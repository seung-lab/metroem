import torch
import time
import numpy as np
import six

from metroem import helpers
from metroem.loss import unsupervised_loss
from torch.cuda.amp import autocast

def combine_pre_post(res, pre, post):
    result = post.from_pixels()(res.from_pixels()(pre.from_pixels())).pixels()
    return result

def combine_pre_post_tgt(res, pre, post, src):
    result_standard = post.from_pixels()(res.from_pixels()(pre.from_pixels()))
    tgt = result_standard(src)
    result = result_standard.pixels()
    return result, tgt

def optimize_pre_post_ups(src, tgt, initial_res, sm, lr, num_iter,
                      src_defects,
                      tgt_defects,
                      opt_params={},
                      opt_mode='adam',
                      crop=16,
                      opt_res_coarsness=0, wd=1e-3, l2=0.0,
                      gridsample_mode="bilinear",
                      sm_keys_to_apply={},
                      mse_keys_to_apply={}):

    opti_loss = unsupervised_loss(smoothness_factor=sm, use_defect_mask=True,
                              sm_keys_to_apply=sm_keys_to_apply,
                              mse_keys_to_apply=mse_keys_to_apply
                          )
    sdb = False
    pred_res = initial_res.clone().detach().field()
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

    loss_bundle = {
        'src': src,
        'tgt': tgt,
        'tgt_defects': tgt_defects,
    }
    loss_bundle['src_defects'] = src_defects
    prev_loss = []

    s = time.time()
    with torch.no_grad():
        loss_bundle = helpers.normalize_bundle(loss_bundle, per_feature_var=True, mask_fill=0)

    loss_bundle['pred_res'] = combine_pre_post(pred_res, pre_res, post_res)

    loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)

    loss_dict = opti_loss(loss_bundle, crop=crop)
    best_loss = loss_dict['result'].cpu().detach().numpy()
    new_best_ago = 0
    lr_halfed_count = 0
    nan_count = 0
    no_impr_count = 0
    new_best_count = 0
    print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())

    for epoch in range(num_iter):
        loss_bundle['pred_res'] = combine_pre_post(pred_res, pre_res, post_res)
        loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
        if sdb:
            import pdb; pdb.set_trace()
        loss_dict = opti_loss(loss_bundle, crop=crop)
        loss_var = loss_dict['result']
        #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        loss_var += (loss_bundle['pred_res']**2).mean() * l2
        curr_loss = loss_var.cpu().detach().numpy()
        #import pdb; pdb.set_trace()

        #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        if np.isnan(curr_loss):
            if sdb:
                print ("NAN LOSS")
                import pdb; pdb.set_trace()
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
                no_impr_count += 1
                if new_best_ago > 12:
                    #print ("No improvement, reducing lr")
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

                optimizer.zero_grad()
                loss_var.backward()
                #torch.nn.utils.clip_grad_norm([pre_res, post_res], 4e0)
                pre_res.grad[pre_res.grad != pre_res.grad] = 0
                post_res.grad[post_res.grad != post_res.grad] = 0
                optimizer.step()
            if lr_halfed_count >= 15 or nan_count > 15:
                break


    loss_bundle['pred_res'] = combine_pre_post(pred_res, prev_pre_res, prev_post_res)
    loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
    loss_dict = opti_loss(loss_bundle, crop=crop)

    e = time.time()
    print ("New best: {}, No impr: {}, Lr halfed: {},  NaN: {}, Iter: {}".format(new_best_count, no_impr_count, lr_halfed, nan_count, epoch))
    print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
    print (e - s)
    print ('==========')


    return loss_bundle['pred_res']

def optimize_pre_post_ups_amp(src, tgt, initial_res, sm, lr, num_iter,
                      src_defects,
                      tgt_defects,
                      opt_params={},
                      opt_mode='adam',
                      crop=16,
                      opt_res_coarsness=0, wd=1e-3, l2=0.0,
                      gridsample_mode="bilinear",
                      sm_keys_to_apply={},
                      mse_keys_to_apply={}):
    opti_loss = unsupervised_loss(smoothness_factor=sm, use_defect_mask=True,
                              sm_keys_to_apply=sm_keys_to_apply,
                              mse_keys_to_apply=mse_keys_to_apply
                          )
    sdb = False
    pred_res = initial_res.clone().detach().field()
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

    loss_bundle = {
        'src': src,
        'tgt': tgt,
        'tgt_defects': tgt_defects,
    }
    loss_bundle['src_defects'] = src_defects
    prev_loss = []

    s = time.time()
    with torch.no_grad():
        loss_bundle = helpers.normalize_bundle(loss_bundle, per_feature_var=True, mask_fill=0)

    loss_bundle['pred_res'] = combine_pre_post(pred_res, pre_res, post_res)

    loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)

    loss_dict = opti_loss(loss_bundle, crop=crop)
    best_loss = loss_dict['result'].cpu().detach().numpy()
    new_best_ago = 0
    lr_halfed_count = 0
    nan_count = 0
    no_impr_count = 0
    new_best_count = 0
    print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())

#        import pdb; pdb.set_trace()

    forward_field_t = 0
    forward_loss_t= 0
    forward_t = 0
    backward_t = 0
        

    for epoch in range(num_iter):
        ft_s = time.time()

        
#        loss_bundle['pred_res'], loss_bundle['pred_tgt'] = combine_pre_post_tgt(pred_res, pre_res, post_res, src)
        loss_bundle['pred_res'] = combine_pre_post(pred_res, pre_res, post_res)
        loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
        if sdb:
            import pdb; pdb.set_trace()
        ft_e = time.time()
        forward_field_t += (ft_e - ft_s)
        ft_s = time.time()
        loss_dict = opti_loss(loss_bundle, crop=crop)
        ft_e = time.time()
        forward_loss_t+= (ft_e - ft_s)
        ft_s = time.time()
        loss_var = loss_dict['result']
        #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())
        if l2 != 0:
            loss_var += (torch.pow(loss_bundle['pred_res'],2)).mean() * l2
        curr_loss = loss_var.cpu().detach().numpy()
        #import pdb; pdb.set_trace()

        #print (loss_dict['result'].cpu().detach().numpy(), loss_dict['similarity'].detach().cpu().numpy(), loss_dict['smoothness'].detach().cpu().numpy())

        ft_e = time.time()
        forward_t += (ft_e - ft_s)
        bt_s = time.time()
        if np.isnan(curr_loss):
            if sdb:
                print ("NAN LOSS")
                import pdb; pdb.set_trace()
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
                no_impr_count += 1
                if new_best_ago > 12:
                    #print ("No improvement, reducing lr")

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

            optimizer.zero_grad()
            loss_var.backward()
            #torch.nn.utils.clip_grad_norm([pre_res, post_res], 4e0)
            pre_res.grad[pre_res.grad != pre_res.grad] = 0
            post_res.grad[post_res.grad != post_res.grad] = 0
            optimizer.step()
            print(curr_loss)
            if lr_halfed_count >= 15 or nan_count > 15:
                break
        bt_e = time.time()
        backward_t += (bt_e - bt_s)


    loss_bundle['pred_res'] = combine_pre_post(pred_res, prev_pre_res, prev_post_res)
    loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(src)
    loss_dict = opti_loss(loss_bundle, crop=crop)

    e = time.time()
    print ("New best: {}, No impr: {}, Lr halfed: {},  NaN: {}, Iter: {}".format(new_best_count, no_impr_count, lr_halfed_count, nan_count, epoch+1))
    print (f"Total loss: {loss_dict['result'].cpu().detach().numpy()}, Similarity: {loss_dict['similarity'].detach().cpu().numpy()}, Smoothness: {loss_dict['smoothness'].detach().cpu().numpy()}")
    print (e - s)
    print ('==========')
    print (f'Benchmarking: forward field {forward_field_t} seconds, forward loss {forward_loss_t} seconds, forward rest {forward_t} seconds, backward pass {backward_t} seconds')


    return loss_bundle['pred_res']
