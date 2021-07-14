import torch
import torchfields

import scipy
import numpy as np
import skimage
import h5py
import time
from skimage import feature
from metroem.helpers import get_np

from scipy.ndimage import convolve
from scipy.ndimage.measurements import label

from pdb import set_trace as st

def get_warped_mask_set(bundle, res, keys_to_apply, inv=False):
    prewarp_result = torch.ones((1, bundle['src'].shape[-2], bundle['src'].shape[-1]),
                        device=bundle['src'].device)
    res = res.squeeze()
    for settings in keys_to_apply:
        name = settings['name']
        if name in bundle:
            mask = bundle[name].squeeze()
            if 'fm' in settings and len(mask.shape) > 2:
                mask = mask[settings['fm']:settings['fm']+1]

            while len(mask.shape) < len(prewarp_result.shape):
                mask = mask.unsqueeze(0)

            if 'binarization' in settings:
                mask = binarize(mask, settings['binarization'])

            if 'mask_value' in settings:
                prewarp_result[mask != 1.0] = settings['mask_value']
            else:
                prewarp_result[mask != 1.0] = 0.0

            around_the_mask = mask
            if 'coarsen_ranges' in settings:
                for length, weight in settings['coarsen_ranges']:
                    if length > 10:
                        around_the_mask = coarsen_mask(around_the_mask, length)
                    else:
                        around_the_mask = coarsen_mask(around_the_mask, length)
                    prewarp_result[(around_the_mask != 1.0) * (prewarp_result == 1)] = weight

    if (res != 0).sum() > 0 and not inv:
        if res.shape[1] == 2:
            result = res(prewarp_result.float())
        else:
            result = res.from_pixels()(prewarp_result.float())
    else:
        result = prewarp_result
    return result


def binarize(img, bin_setting):
    if bin_setting['strat']== 'eq':
        result = (img == bin_setting['value'])
    elif bin_setting['strat']== 'neq':
        result = (img != bin_setting['value'])
    elif bin_setting['strat']== 'lt':
        result = (img < bin_setting['value'])
    elif bin_setting['strat'] == 'gt':
        result = (img > bin_setting['value'])
    elif bin_setting['strat']== 'between':
        result = ((img > bin_setting['range'][0]) * (img < bin_setting['range'][0]))
    elif bin_setting['strat']== 'not_between':
        result = ((img < bin_setting['range'][0]) + (img > bin_setting['range'][0])) > 0
    return result.float()


def get_mse_and_smoothness_masks(bundle, mse_keys_to_apply,
                                  sm_keys_to_apply, **kwargs):
    if 'src' not in mse_keys_to_apply:
        mse_keys_to_apply['src'] = []
    if 'tgt' not in mse_keys_to_apply:
        mse_keys_to_apply['tgt'] = []
    if 'src' not in sm_keys_to_apply:
        sm_keys_to_apply['src'] = []
    if 'tgt' not in sm_keys_to_apply:
        sm_keys_to_apply['tgt'] = []
    return get_warped_srctgt_mask(bundle, mse_keys_to_apply, sm_keys_to_apply)


def get_warped_srctgt_mask(bundle, mse_keys_to_apply, sm_keys_to_apply):
    src_shape = bundle['src'].shape
    if len(src_shape) == 4:
        mask_shape = (src_shape[0], 1, src_shape[2], src_shape[3])
        num_channels = src_shape[1]
        channel_dim = 1
    elif len(src_shape) == 3:
        mask_shape = (1, src_shape[1], src_shape[2])
        num_channels = src_shape[0]
        channel_dim = 0
    else:
        raise Exception(f"Unsupported image shape: {src_shape}")

    mask_mse = torch.ones(mask_shape, device=bundle['src'].device)
    mask_sm  = torch.ones(mask_shape, device=bundle['src'].device)
    pred_res = bundle['pred_res']

    if 'src' in mse_keys_to_apply:
        src_mask_mse = get_warped_mask_set(bundle, pred_res,
                                       mse_keys_to_apply['src'])
        mask_mse *= src_mask_mse

    if 'tgt' in mse_keys_to_apply:
        tgt_mask_mse = get_warped_mask_set(bundle, torch.zeros_like(pred_res),
                                       mse_keys_to_apply['tgt'])
        mask_mse *= tgt_mask_mse

    if 'src' in sm_keys_to_apply:
        src_mask_sm = get_warped_mask_set(bundle, pred_res,
                                       sm_keys_to_apply['src'])
        mask_sm *= src_mask_sm

    if 'tgt' in sm_keys_to_apply:
        tgt_mask_sm = get_warped_mask_set(bundle, torch.zeros_like(pred_res),
                                       sm_keys_to_apply['tgt'])
        mask_sm *= tgt_mask_sm

    if 'src_tgt_comb' in sm_keys_to_apply:
        src_comb_mask_sm = get_warped_mask_set(bundle, torch.zeros_like(pred_res),
                                       sm_keys_to_apply['src_tgt_comb']['src'])
        tgt_comb_mask_sm = get_warped_mask_set(bundle, torch.zeros_like(pred_res),
                                       sm_keys_to_apply['src_tgt_comb']['tgt'])
        mask_sm *= ((tgt_comb_mask_sm + src_comb_mask_sm) > 0).float()

    mask_mse = torch.cat([mask_mse] * num_channels, channel_dim)

    return mask_mse, mask_sm


RAW_WHITE_THRESHOLD = -0.485
def get_raw_defect_mask(img, threshold=-1):
    result = 1 - ((img < threshold) * (img > RAW_WHITE_THRESHOLD))
    return result.type(torch.cuda.FloatTensor)

def get_raw_white_mask(img):
    result = img >= RAW_WHITE_THRESHOLD
    return result.type(torch.cuda.FloatTensor)

def get_defect_mask(img, threshold=-3.5):
    result = 1 - ((img < threshold) * (img > -3.9999))
    return result.type(torch.cuda.FloatTensor)

def get_brightness_mask(img, low_cutoff, high_cutoff):
    result = (img >= low_cutoff)* (img <= high_cutoff)
    return result.type(torch.cuda.FloatTensor)

def get_blood_vessel_mask(img, threshold=2.5):
    result = img >= threshold
    return result.type(torch.cuda.FloatTensor)

def get_white_mask(img, threshold=-3.5):
    result = img >= threshold
    return result.type(torch.cuda.FloatTensor)

def get_black_mask(img):
    result = img < 3.55
    return result.type(torch.cuda.FloatTensor)


# Numpy masks
def get_very_white_mask(img):
    # expects each pixel in range [-0.5, 0.5]
    # used at mip 8
    return img > 0.04

def coarsen_mask(mask, n=1, flip=True):
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    for _ in range(n):
        if isinstance(mask, np.ndarray):
            mask = convolve(mask, kernel) > 0
            mask = mask.astype(np.int16) > 1
        else:
            if mask.device == torch.device('cpu'):
                kernel_var = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(mask.device)
            else:
                kernel_var = torch.cuda.FloatTensor(kernel).unsqueeze(0).unsqueeze(0).to(mask.device)
            k = torch.nn.Parameter(data=kernel_var, requires_grad=False)
            if flip:
                mask = mask.logical_not().float()
            while len(mask.shape) < 4:
                mask = mask.unsqueeze(0)
            mask =  (torch.nn.functional.conv2d(mask,
                kernel_var, padding=1) > 1).squeeze(1)
            if flip:
                mask = mask.logical_not()
            mask = mask.float()
    return mask


def dilate(a):
    return scipy.ndimage.morphology.binary_dilation(a != 0)
    conn = scipy.ndimage.generate_binary_structure(2, 2)
    return scipy.ndimage.morphology.binary_dilation(a, conn)


def erode(a):
    return scipy.ndimage.morphology.binary_erosion(a != 0)


def closing(a, n=2):
    result = a
    for _ in range(n):
        result = dilate(result != 0)
    for _ in range(n):
        result = erode(result != 0)
    return result
