import torch
import time
import numpy as np
import six
import cc3d
import fastremap
from collections import defaultdict

from metroem import helpers
from metroem.loss import unsupervised_loss
from metroem import masks


def get_aligned_mask(tgt, pred_tgt, sensitivity=0.5):
    tgt = helpers.get_np(tgt)
    pred_tgt = helpers.get_np(pred_tgt)
    diff = np.abs(helpers.get_np((tgt - pred_tgt)))
    bad_pixels = diff > np.abs(tgt).max() * (1 - sensitivity)

    bad_pixels_d = bad_pixels
    for _ in range(2):
        bad_pixels_d = masks.dilate(bad_pixels_d)
    tgt_tissue = helpers.get_np(tgt != 0)
    pred_tgt_tissue = helpers.get_np(pred_tgt != 0)
    good_tissue = tgt_tissue * pred_tgt_tissue * (bad_pixels_d == 0)
    return good_tissue

def get_nailed_region(tgt, pred_tgt, sensitivity=0.5):
    aligned_mask = get_aligned_mask(tgt, pred_tgt, sensitivity)
    cc_labels = cc3d.connected_components(aligned_mask)
    segids, counts = np.unique(cc_labels, return_counts=True)
    segids = [ segid for segid, ct in zip(segids, counts) if ct > 50 ]
    filtered_mask = fastremap.mask_except(cc_labels, segids, in_place=True) != 0
    filtered_mask = masks.closing(filtered_mask, n=4)
    return filtered_mask

def combine_pre_post(res, pre, post):
    result = post.from_pixels()(res.from_pixels()(pre.from_pixels())).pixels()
    return result

def optimize_pre_post_ups(src, tgt, initial_res, sm, lr, num_iter,
                      src_defects,
                      tgt_defects,
                      opt_params={},
                      opt_mode='adam',
                      crop=16,
                      noimpr_period=50,
                      opt_res_coarsness=0, wd=1e-4, l2=1e-4,
                      gridsample_mode="trilinear",
                      normalize=True,
                      optimize_init=False,
                      sm_keys_to_apply={},
                      mse_keys_to_apply={},
                      verbose=True,
                      mask_around_nailed=False,
                      nailed_check_period=None,
                      max_bad=15

                    ):

    timings = defaultdict(float)
    ts = time.time()
    opti_loss = unsupervised_loss(
          smoothness_factor=sm, use_defect_mask=True,
          sm_keys_to_apply=sm_keys_to_apply,
          mse_keys_to_apply=mse_keys_to_apply
      )
    pred_res = initial_res.clone().detach().field()
    pred_res = pred_res.down(opt_res_coarsness)
    if optimize_init:
        pred_res.requires_grad = True
    else:
        pred_res.requires_grad = False

    pre_res = torch.zeros_like(pred_res, device=pred_res.device).field().detach()
    post_res = torch.zeros_like(pre_res, device=pred_res.device).field().detach()

    prev_pre_res = torch.zeros_like(pre_res, device=pred_res.device).field().detach()
    prev_post_res = torch.zeros_like(pre_res, device=pred_res.device).field().detach()

    for t in [pre_res, post_res, prev_pre_res, prev_post_res]:
        t.requires_grad = True
    trainable = [pre_res, post_res]
    if opt_mode == 'adam':
        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=wd)
    elif opt_mode == 'sgd':
        optimizer = torch.optim.SGD(trainable, lr=lr, **opt_params)

    if normalize:
        with torch.no_grad():
            src_mask = torch.logical_not(src_defects)
            tgt_mask = torch.logical_not(tgt_defects)

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
    }

    prev_loss = []
    s = time.time()

    # last_nailed = 0
    # original_tgt = tgt
    # original_tgt_zeros = helpers.get_np(original_tgt[0, 0] == 0)
    # nail_mask = None

    if normalize:
        with torch.no_grad():
            pass
            #loss_bundle = helpers.normalize_bundle(loss_bundle, per_feature_var=True, mask_fill=0)

    loss_bundle['pred_res'] = combine_pre_post(pred_res, pre_res, post_res).up(opt_res_coarsness)

    loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
    loss_dict = opti_loss(loss_bundle, crop=crop)
    best_loss = loss_dict['result'].cpu().detach().numpy()
    new_best_ago = 0
    lr_halfed_count = 0
    nan_count = 0
    no_impr_count = 0
    new_best_count = 0

    if verbose:
        print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())

    timings["01 Preparation"] += (time.time() - ts)
    prof = None
    with torch.autograd.profiler.profile(enabled=False, use_cpu=True, use_cuda=True, with_stack=True) as prof:
        for epoch in range(num_iter):
            ts = time.time()
            loss_bundle['pred_res'] = combine_pre_post(pred_res, pre_res, post_res).up(opt_res_coarsness)
            loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
            timings["02 Combine Pre Post"] += (time.time() - ts)

            ts = time.time()
            loss_dict = opti_loss(loss_bundle, crop=crop)
            loss_var = loss_dict['result']
            #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
            loss_var += (loss_bundle['pred_res']**2).mean() * l2
            curr_loss = loss_var.cpu().detach().numpy()
            timings["03 Opti Loss"] += (time.time() - ts)

            #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
            ts = time.time()
            if np.isnan(curr_loss):
                nan_count += 1
                lr /= 1.5
                lr_halfed_count += 1
                pre_res = prev_pre_res.clone().detach()
                post_res = prev_post_res.clone().detach()
                post_res.requires_grad = True
                pre_res.requires_grad = True
                trainable = [pre_res, post_res]
                if opt_mode == 'adam':
                    optimizer = torch.optim.Adam([pre_res, post_res], lr=lr, weight_decay=wd)
                elif opt_mode == 'sgd':
                    optimizer = torch.optim.SGD([pre_res, post_res], lr=lr, **opt_params)
                prev_loss = []
                new_best_ago = 0
                timings["04 Optimizer Setup"] += (time.time() - ts)
            else:
                min_improve = 1e-11
                if not np.isnan(curr_loss) and curr_loss + min_improve <= best_loss:
                    prev_pre_res = pre_res.clone()
                    prev_post_res = post_res.clone()
                    best_loss = curr_loss
                    #print ("new best")
                    new_best_count += 1
                    new_best_ago = 0
                else:
                    new_best_ago += 1
                    if new_best_ago > noimpr_period:
                        #print ("No improvement, reducing lr")
                        no_impr_count += 1
                        lr /= 2
                        lr_halfed_count += 1
                        #pre_res = prev_pre_res.clone().detach()
                        #post_res = prev_post_res.clone().detach()
                        #post_res.requires_grad = True
                        #pre_res.requires_grad = True
                        if opt_mode == 'adam':
                            optimizer = torch.optim.Adam([pre_res, post_res], lr=lr)
                        elif opt_mode == 'sgd':
                            optimizer = torch.optim.SGD([pre_res, post_res], lr=lr, **opt_params)
                        new_best_ago -= 5
                    prev_loss.append(curr_loss)
                timings["04 Optimizer Setup"] += (time.time() - ts)
                ts = time.time()
                optimizer.zero_grad()
                loss_var.backward()
                #torch.nn.utils.clip_grad_norm([pre_res, post_res], 4e0)
                torch.nan_to_num_(pre_res.grad, 0)
                torch.nan_to_num_(post_res.grad, 0)
                optimizer.step()
                timings["05 Backward Step"] += (time.time() - ts)

                if lr_halfed_count >= max_bad or nan_count > max_bad:
                    break

    if prof is not None:
        with open("timings3.log", "a") as f:
            f.write(str(prof.key_averages(group_by_stack_n=7).table(sort_by="cuda_time_total", top_level_events_only=True)))
            f.write("\n*******************************************\n\n")

    ts = time.time()
    loss_bundle['pred_res'] = combine_pre_post(pred_res, prev_pre_res, prev_post_res).up(opt_res_coarsness)

    loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
    loss_dict = opti_loss(loss_bundle, crop=crop)
    timings["06 Final Step"] += (time.time() - ts)

    print("\n")
    for k, v in timings.items():
        print(f"{k}: {round(v, 4)}")
    print("\n")

    e = time.time()
    if verbose:
        print ("New best: {}, No impr: {}, NaN: {}, Iter: {}".format(new_best_count, no_impr_count, nan_count, epoch))
        print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        print (e - s)
        print ('==========')

    print(1000*(e-s) / epoch, "ms per epoch")

    # exit(0)


    return loss_bundle['pred_res']
