import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import six
import copy
from collections import defaultdict

from pdb import set_trace as st

from metroem import helpers
from metroem.masks import get_mse_and_smoothness_masks
from metroem.alignment import align_sample

def lap(fields):
    def dx(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,1:-1,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,1:-1,:] - f[:,:,:-2,:], p), 2)
    def dxf(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,1:-1,:,:] - f[:,2:,:,:], p), 1)
    def dyf(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,1:-1,:] - f[:,:,2:,:], p), 2)
    fields = map(lambda f: [dx(f), dy(f), dxf(f), dyf(f)], fields)
    fields = map(lambda fl: (sum(fl) / 4.0) ** 2, fields)
    field = sum(map(lambda f: torch.sum(f, -1), fields))
    return field


def jacob(fields):
    def dx(f):
        p = Variable(torch.zeros((f.size(0),1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((f.size(0),f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.cat(fields, -1) ** 2, -1)
    return field


def cjacob(fields):
    def center(f):
        fmean_x, fmean_y = torch.mean(f[:,:,:,0]).data[0], torch.mean(f[:,:,:,1]).data[0]
        fmean = torch.cat((fmean_x * torch.ones((1,f.size(1), f.size(2),1)), fmean_y * torch.ones((1,f.size(1), f.size(2),1))), 3)
        fmean = Variable(fmean).cuda()
        return f - fmean

    def dx(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        d = torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
        return center(d)
    def dy(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        d = torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
        return center(d)

    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.cat(fields, -1) ** 2, -1)
    return field


def njacob(fields):
    f = fields[0]
    f2 = torch.tensor(f, requires_grad=True)
    f2[:, :, :, 0] = f[:, :, :, 0] / torch.mean(torch.abs(f[:, :, :, 0]))
    f2[:, :, :, 1] = f[:, :, :, 1] / torch.mean(torch.abs(f[:, :, :, 1]))
    return jacob([f2])


def tv(fields):
    def dx(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,2:,:,:] - f[:,:-2,:,:], p), 1)
    def dy(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,2:,:] - f[:,:,:-2,:], p), 2)
    fields = sum(map(lambda f: [dx(f), dy(f)], fields), [])
    field = torch.sum(torch.abs(torch.cat(fields, -1)), -1)
    return field


def field_dx(f, forward=False):
    if forward:
        delta = f[:,1:-1,:,:] - f[:,2:,:,:]
    else:
        delta = f[:,1:-1,:,:] - f[:,:-2,:,:]
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 0, 0, 1, 1, 0, 0))
    return result

def field_dy(f, forward=False):
    if forward:
        delta = f[:,:,1:-1,:] - f[:,:,2:,:]
    else:
        delta = f[:,:,1:-1,:] - f[:,:,:-2,:]
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 0, 0, 0, 0))
    return result

def field_dxy(f, forward=False):
    if forward:
        delta = f[:,1:-1,1:-1,:] - f[:,2:,2:,:]
    else:
        delta = f[:,1:-1,1:-1,:] - f[:,:-2,:-2,:]

    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 1, 1, 0, 0))
    return result

def field_dxy2(f, forward=False):
    if forward:
        delta = f[:, 1:-1, 1:-1, :] - f[:, 2:, :-2, :]
    else:
        delta = f[:, 1:-1, 1:-1, :] - f[:, :-2, 2:, :]

    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 1, 1, 0, 0))
    return result

def rigidity_score(field_delta, tgt_length, power=2):
    spring_lengths = torch.sqrt(field_delta[..., 0]**2 + field_delta[..., 1]**2 + 1e-8)
    spring_deformations = (spring_lengths - tgt_length).abs() ** power
    return spring_deformations

def pix_identity(size, batch=1, device='cuda'):
    result = torch.zeros((batch, size, size, 2), device=device)
    x = torch.arange(size, device=device)
    result[:, :, :, 0] = x
    result = torch.transpose(result, 1, 2)
    result[:, :, :, 1] = x
    result = torch.transpose(result, 1, 2)
    return result

def rigidity(field, power=2, diagonal_mult=1.0):
    # Kernel on Displacement field yields change of displacement
    diff_ker = torch.tensor([
        [
          [[ 0, 0, 0],
           [-1, 1, 0],
           [ 0, 0, 0]],

          [[ 0,-1, 0],
           [ 0, 1, 0],
           [ 0, 0, 0]],

          [[-1, 0, 0],
           [ 0, 1, 0],
           [ 0, 0, 0]],

          [[ 0, 0,-1],
           [ 0, 1, 0],
           [ 0, 0, 0]],
        ]
    ], dtype=field.dtype, device=field.device)

    diff_ker = diff_ker.permute(1, 0, 2, 3).repeat(2, 1, 1, 1)

    # Add distance between pixel to get absolute displacement
    diff_bias = torch.tensor(
        [1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 1.0],
        dtype=field.dtype,
        device=field.device,
    )
    delta = torch.conv2d(field, diff_ker, diff_bias, groups=2, padding=[2, 2])
    delta = delta.reshape(2, 4, *delta.shape[-2:]).permute(1, 2, 3, 0)

    spring_lengths = torch.norm(delta, dim=3)

    spring_defs = torch.stack([
        spring_lengths[0, 1:-1, 1:-1] - 1,
        spring_lengths[0, 1:-1, 2:  ] - 1,

        spring_lengths[1, 1:-1, 1:-1] - 1,
        spring_lengths[1, 2:  , 1:-1] - 1,

        (spring_lengths[2, 1:-1, 1:-1] - 2**(1/2)) * (diagonal_mult)**(1/power),
        (spring_lengths[2, 2:  , 2:  ] - 2**(1/2)) * (diagonal_mult)**(1/power),

        (spring_lengths[3, 1:-1, 1:-1] - 2**(1/2)) * (diagonal_mult)**(1/power),
        (spring_lengths[3, 2: ,  0:-2] - 2**(1/2)) * (diagonal_mult)**(1/power),
    ])

    # Slightly faster than sum() + pow(), and no need for abs() if power is odd
    result = torch.norm(spring_defs, p=power, dim=0).pow(power)

    total = 4 + 4 * diagonal_mult

    result /= total

    # Remove incorrect smoothness values caused by 2px zero padding
    result[..., 0:2, :] = 0
    result[..., -2:, :] = 0
    result[..., :, 0:2] = 0
    result[..., :, -2:] = 0

    return result.squeeze()

def smoothness_penalty(ptype='jacob'):
    def penalty(fields, weights=None):
        if ptype ==     'lap': field = lap(fields)
        elif ptype == 'jacob': field = jacob(fields)
        elif ptype == 'cjacob': field = cjacob(fields)
        elif ptype == 'njacob': field = njacob(fields)
        elif ptype ==    'tv': field = tv(fields)
        elif ptype == 'rig': field = rigidity(fields[0])
        elif ptype == 'linrig': field = rigidity(fields[0], power=1)
        else: raise ValueError("Invalid penalty type: {}".format(ptype))

        #if weights is not None:
        #    field = field * weights
        return field
    return penalty


def get_dataset_loss(model, dataset_loader, loss_fn, mip_in, *args, **kwargs):
    losses = {}
    losses['result'] = []
    for sample in dataset_loader:
        aligned_bundle = align_sample(model, sample)
        if aligned_bundle is not None:
            loss_result = loss_fn(aligned_bundle)
            if isinstance(loss_result, dict):
                for k, v in six.iteritems(loss_result):
                    if isinstance(loss_result[k], torch.Tensor):
                        if k not in losses:
                            losses[k] = []
                        losses[k].append(loss_result[k].cpu().detach().numpy())
            else:
                losses['result'].append(loss_result.cpu().detach().numpy())
    for k in losses.keys():
        losses[k] = np.average(losses[k])
    return losses

def similarity_score(bundle, weights=None, crop=32):
    tgt = bundle['tgt']
    pred_tgt = bundle['pred_tgt']
    mse = torch.pow(tgt - pred_tgt, 2)
    if crop > 0:
        mse = mse[..., crop:-crop, crop:-crop]
    if weights is not None:
        weights = weights
        if crop > 0:
            weights = weights[..., crop:-crop, crop:-crop]
        mean_mse = torch.mean(mse * weights)
    else:
        mean_mse = torch.mean(mse)
    return mean_mse

def smoothness_score(bundle, smoothness_type,
                     weights=None, crop=8):
    pixelwise = smoothness_penalty(smoothness_type)([bundle['pred_res']])
    if crop > 0:
        pixelwise = pixelwise[..., crop:-crop, crop:-crop]

    if weights is not None:
        weights = weights
        if crop > 0:
            weights = weights[..., crop:-crop, crop:-crop]
        mean_sm = torch.mean(pixelwise * weights)
    else:
        mean_sm = torch.mean(pixelwise)
    return mean_sm


def similarity_sampling_loss(sample_size, unsup_loss, sample_coverage=0.001, min_nonblack=0.5):
    downsampler = torch.nn.functional.avg_pool2d

    def loss_fn(bundle):
        mse_mask = unsup_loss(bundle, smoothness_mult=0)['mse_mask']
        diff = (bundle['src'] - bundle['tgt'])**2
        mse_mask_tiles = downsampler(mse_mask, sample_size)
        diff_tiles = downsampler(diff * mse_mask, sample_size)
        result = diff_tiles / (mse_mask_tiles + 1e-8)
        result[mse_mask_tiles < min_nonblack] = 0
        #defec_tile_ids = (mse_mask_tiles > min_nonblack) * (mse_mask_tiles)
        #defect_tiles = result[result[(mse_mask_tiles > min_nonblack) * mse_mask_tiles ]]
        good_tiles = result[result != 0]
        if len(good_tiles) < 10:
            return {'similarity': 0}
        median = np.median(good_tiles.detach().cpu().numpy())
        lower_cutoff = np.percentile(good_tiles.detach().cpu().numpy(), 15)
        sampled_result = good_tiles[(good_tiles > lower_cutoff) * (good_tiles < median)].mean()
        '''tile_num = len(good_tiles)

        sample_num = int(tile_num * sample_coverage)
        if sample_num == 0:
            return {'similarity': 0}
        actual_coverage = sample_num / tile_num
        sample_result = 0
        perm = torch.randperm(good_tiles.size(0))
        idx = perm[:sample_num]
        chosen_samples = good_tiles[idx]
        sampled_result = chosen_samples.mean()
        '''
        #print ("{} -> {}".format(diff[mse_mask > 0].mean(), sampled_result))
        return {'similarity': sampled_result}


    return loss_fn


def multilevel_metric_loss(levels, mip_in, loss_fn,
        norm_embeddings=True, sm_div=False, train_thru=True,
        pre_align_sample=None, pure_emb=False, pre_align_weight=0.7):

    def compute_loss(loss_bundle, crop=32):
        loss_dict = defaultdict(lambda : 0)
        state = loss_bundle['state']


        for l in levels:
            if pre_align_sample is None:
                pre_align_loss_fn = loss_fn
            else:
                pre_align_loss_fn = similarity_sampling_loss(
                        sample_size=pre_align_sample,
                        unsup_loss=loss_fn)

            loss_bundle_emb = copy.copy(loss_bundle)
            embedding = state['up'][str(l)]['skip']

            if not train_thru:
                embedding = embedding.clone().data

            num_features = embedding.shape[1]
            loss_bundle_emb['src'] = embedding[:, 1:num_features//2]
            loss_bundle_emb['tgt'] = embedding[:, 1+num_features//2:]
            for _ in range(l - mip_in):
                for k in loss_bundle.keys():
                    if 'field' in k or 'res' in k:
                        loss_bundle_emb[k] = loss_bundle_emb[k].from_pixels().down().pixels()
                    elif 'src_' in k or 'tgt_' in k:
                        if loss_bundle_emb[k].dtype == torch.bool:
                            loss_bundle_emb[k] = torch.nn.functional.max_pool2d(loss_bundle_emb[k].float(), 2).bool()
                        else:
                            loss_bundle_emb[k] = torch.nn.functional.max_pool2d(loss_bundle_emb[k], 2)

            if norm_embeddings:
                with torch.set_grad_enabled(True):
                    loss_bundle_emb = helpers.normalize_bundle(loss_bundle_emb, per_feature_var=True, mask_fill=0)

            loss_bundle_emb['pred_tgt'] = loss_bundle_emb['src']
            #if 'src_field' in loss_bundle_emb:
            #    loss_bundle_emb['pred_tgt'] = \
            #            loss_bundle_emb['src_field'].field().from_pixels()(loss_bundle_emb['src'])

            if pure_emb:
                auger = metroem.augmentations.RandomWarp(difficulty=2, max_disp=50, min_disp=-50)
                aug_loss_bundle_emb = auger(copy.deepcopy(loss_bundle_emb))
                pre_align_loss =  pre_align_loss_fn(loss_bundle_emb)['similarity']
                print ('1', pre_align_loss)
                pre_align_loss =  pre_align_loss_fn(aug_loss_bundle_emb)['similarity']
                print ('2', pre_align_loss)
            else:
                pre_align_loss =  pre_align_loss_fn(loss_bundle_emb)['similarity']

            loss_dict['result'] = loss_dict['result'] -  pre_align_weight * pre_align_loss / len(levels)

            if sm_div:
                sm_mult = 1.0 / 2**(l - mip_in)
            else:
                sm_mult = 1.0

            loss_bundle_emb['pred_tgt'] = loss_bundle_emb['pred_res'].from_pixels()(loss_bundle_emb['src'])
            level_loss_dict = loss_fn(loss_bundle_emb, smoothness_mult=sm_mult, crop=crop)
            loss_dict['result'] = loss_dict['result'] + level_loss_dict['result'] / len(levels)
            loss_dict['similarity'] = loss_dict['similarity'] + level_loss_dict['similarity'] / len(levels)
            if loss_dict['vec_magnitude'] == 0:
                loss_dict['vec_magnitude'] = level_loss_dict['vec_magnitude']

            loss_dict['smoothness'] = loss_dict['smoothness'] + level_loss_dict['smoothness'] / len(levels)
            #print ("Level {} posst alignment: sim {:.2E}, sm {:.2E}".format(l, level_loss_dict['similarity'], float(level_loss_dict['smoothness'])))
            #print ("Mean: {:.2E}, Var: {:.2E}".format(
            #loss_bundle_emb['src'].mean(),
            #loss_bundle_emb['src'].var()
            #    ))
        return loss_dict

    return compute_loss

def unsupervised_loss(smoothness_factor, smoothness_type='rig', use_defect_mask=False,
                      sm_keys_to_apply={}, mse_keys_to_apply={}):
    def compute_loss(bundle, smoothness_mult=1.0, crop=32):
        loss_dict = {}
        if use_defect_mask:
            mse_mask, smoothness_mask = get_mse_and_smoothness_masks(bundle,
                    sm_keys_to_apply=sm_keys_to_apply,
                    mse_keys_to_apply=mse_keys_to_apply)
        else:
            mse_mask = None
            smoothness_mask = None
        similarity = similarity_score(bundle,
                                      weights=mse_mask,
                                      crop=crop)
        if smoothness_mult != 0:
            smoothness = smoothness_score(bundle,
                                      weights=smoothness_mask,
                                      smoothness_type=smoothness_type,
                                      crop=crop)
        else:
            smoothness = torch.zeros(1, device=bundle['src'].device, dtype=torch.float32)
        result =  similarity + smoothness * smoothness_factor

        loss_dict['result'] = result
        loss_dict['similarity'] = similarity
        loss_dict['smoothness'] = smoothness * smoothness_factor * smoothness_mult
        loss_dict['vec_magnitude'] = torch.mean(torch.abs(bundle['pred_res']))
        loss_dict['vec_sim'] = torch.cuda.FloatTensor([0])
        if 'res' in bundle:
            loss_dict['vec_sim'] = torch.mean(torch.abs(bundle['pred_res'] - bundle['res']))
        loss_dict['mse_mask'] = mse_mask
        loss_dict['smoothness_mask'] = smoothness_mask
        return loss_dict
    return compute_loss


