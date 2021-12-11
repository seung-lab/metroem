import torch
import time
import numpy as np
import torchfields
from collections import defaultdict

from metroem import helpers
from metroem.loss import unsupervised_loss


def combine_pre_post(res, post):
    result = post.from_pixels()(res.from_pixels()).pixels()
    return result

def optimize_pre_post_ups(src, tgt, initial_res, sm, lr, num_iter,
                      src_defects,
                      tgt_defects,
                      src_zeros,
                      tgt_zeros,
                      opt_params=None,
                      opt_mode='adam',
                      crop=16,
                      noimpr_period=50,
                      opt_res_coarsness=0,
                      wd=0,
                      l2=1e-4,
                      normalize=True,
                      optimize_init=False,
                      sm_keys_to_apply=None,
                      mse_keys_to_apply=None,
                      verbose=False,
                      max_bad=15
                    ):
    loss_history = defaultdict(lambda: [])
    if opt_params is None:
        opt_params = {}
    if sm_keys_to_apply is None:
        sm_keys_to_apply = {}
    if mse_keys_to_apply is None:
        mse_keys_to_apply = {}

    opti_loss = unsupervised_loss(
          smoothness_factor=sm, use_defect_mask=True,
          sm_keys_to_apply=sm_keys_to_apply,
          mse_keys_to_apply=mse_keys_to_apply
      )
    pred_res = initial_res.detach().field()
    pred_res = pred_res.down(opt_res_coarsness)
    pred_res.requires_grad = optimize_init

    post_res = torchfields.Field.zeros_like(pred_res, device=pred_res.device, requires_grad=True)

    trainable = [post_res]
    if opt_mode == 'adam':
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    elif opt_mode == 'sgd':
        optimizer = torch.optim.SGD(trainable, lr=lr, **opt_params)

    if normalize:
        with torch.no_grad():
            src_mask = torch.logical_not(src_zeros)
            tgt_mask = torch.logical_not(tgt_zeros)

            while src_mask.ndim < src.ndim:
                src_mask.unsqueeze_(0)
            while tgt_mask.ndim < src.ndim:
                tgt_mask.unsqueeze_(0)

            src = helpers.normalize(src, mask=src_mask, mask_fill=0)
            tgt = helpers.normalize(tgt, mask=tgt_mask, mask_fill=0)

    loss_bundle = {
        'src': src,
        'tgt': tgt,
        'src_defects': src_defects,
        'tgt_defects': tgt_defects,
        'src_zeros': src_zeros,
        'tgt_zeros': tgt_zeros,
    }

    prev_loss = []
    s = time.time()

    loss_bundle['pred_res'] = combine_pre_post(pred_res, post_res).up(opt_res_coarsness)
    loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
    loss_dict = opti_loss(loss_bundle, crop=crop)
    best_loss = loss_dict['result'].detach().cpu().numpy()
    new_best_ago = 0
    lr_halfed_count = 0
    no_impr_count = 0
    new_best_count = 0
    if verbose:
        #print (loss_dict['result'].detach().cpu().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        print (loss_dict['result'].detach().cpu().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())

    loss_history['result'].append(loss_dict['result'].detach().cpu().numpy().item())

    for epoch in range(num_iter):
        loss_bundle['pred_res'] = combine_pre_post(pred_res, post_res).up(opt_res_coarsness)
        loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
        loss_dict = opti_loss(loss_bundle, crop=crop)
        loss_var = loss_dict['result']
        loss_history['result'].append(loss_dict['result'].detach().cpu().numpy().item())
        loss_var += (loss_bundle['pred_res']**2).mean() * l2
        curr_loss = loss_var.detach().cpu().numpy()

        min_improve = 1e-11
        if curr_loss + min_improve <= best_loss:
            # Improvement
            best_loss = curr_loss
            new_best_count += 1
            new_best_ago = 0
        else:
            new_best_ago += 1
            if new_best_ago > noimpr_period:
                # No improvement, reduce learning rate
                no_impr_count += 1
                lr /= 2
                lr_halfed_count += 1

                if opt_mode == 'adam':
                    optimizer = torch.optim.Adam([post_res], lr=lr, weight_decay=wd)
                elif opt_mode == 'sgd':
                    optimizer = torch.optim.SGD([post_res], lr=lr, **opt_params)
                new_best_ago -= 5
            prev_loss.append(curr_loss)

        optimizer.zero_grad()
        loss_var.backward()
        optimizer.step()

        if lr_halfed_count >= max_bad:
            break

    loss_bundle['pred_res'] = combine_pre_post(pred_res, post_res).up(opt_res_coarsness)
    loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
    loss_dict = opti_loss(loss_bundle, crop=crop)

    e = time.time()

    if verbose:
        print ("New best: {}, No impr: {}, Iter: {}".format(new_best_count, no_impr_count, epoch))
        print (loss_dict['result'].detach().cpu().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        print (e - s)
        print ('==========')

    return loss_history#loss_bundle['pred_res'].field()
