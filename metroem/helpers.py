import os
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import rescale
from functools import reduce
import subprocess
from torch.nn.parameter import Parameter
from cloudvolume import CloudVolume

import h5py
import artificery
import json

def get_random_crop_coords(full_shape, cropped_shape, coord_granularity=4):
    assert cropped_shape[0] <= full_shape[0]
    assert cropped_shape[1] <= full_shape[1]
    assert cropped_shape[0] % coord_granularity == 0
    assert cropped_shape[1] % coord_granularity == 0

    x_bot_preshift = np.random.randint(0, full_shape[0] - cropped_shape[0] + 1)
    y_bot_preshift = np.random.randint(0, full_shape[1] - cropped_shape[1] + 1)
    x_bot = x_bot_preshift - (x_bot_preshift % coord_granularity)
    y_bot = y_bot_preshift - (y_bot_preshift % coord_granularity)
    x_top = x_bot + cropped_shape[0]
    y_top = y_bot + cropped_shape[1]

    return x_bot, x_top, y_bot, y_top

def get_center_crop_coords(full_shape, cropped_shape, coord_granularity=4):
    assert cropped_shape[0] <= full_shape[0]
    assert cropped_shape[1] <= full_shape[1]

    assert cropped_shape[0] % coord_granularity == 0
    assert cropped_shape[1] % coord_granularity == 0

    x_bot_preshift = (full_shape[0] - cropped_shape[0]) // 2
    y_bot_preshift = (full_shape[1] - cropped_shape[1]) // 2
    x_bot = x_bot_preshift - (x_bot_preshift % coord_granularity)
    y_bot = y_bot_preshift - (y_bot_preshift % coord_granularity)

    x_top = x_bot + cropped_shape[0]
    y_top = y_bot + cropped_shape[1]

    return x_bot, x_top, y_bot, y_top

def random_crop(img, cropped_shape):
    result = []
    if isinstance(img, list):
        original_shape = img[0].shape[-2:]
        x_bot, x_top, y_bot, y_top = get_random_crop_coords(original_shape, cropped_shape)
        for i in img:
            assert (i.shape[-2] == original_shape[-2])
            assert (i.shape[-1] == original_shape[-1])

            result.append(i[..., x_bot:x_top, y_bot:y_top])
    else:
        original_shape = img.shape
        x_bot, x_top, y_bot, y_top = get_random_crop_coords(original_shape, cropped_shape)

        result.append(img[..., x_bot:x_top, y_bot:y_top])
    return result


def open_model(name, checkpoint_folder, device='cpu'):
    a = artificery.Artificery()

    spec_path = os.path.join(checkpoint_folder, "model_spec.json")
    my_p = a.parse(spec_path)

    checkpoint_path = os.path.join(checkpoint_folder, "{}.state.pth.tar".format(name))
    if os.path.isfile(checkpoint_path):
        my_p.load_state_dict(torch.load(checkpoint_path))
    my_p.name = name
    return my_p.to(device)

def create_model(name, model_spec, checkpoint_folder, write_spec=True):
    checkpoint_path = os.path.join(checkpoint_folder, "{}.state.pth.tar".format(name))
    print (checkpoint_path)
    if os.path.isfile(checkpoint_path):
        #my_p.load_state_dict(torch.load(checkpoint_path))
        a = artificery.Artificery(checkpoint_init=False)
    else:
        a = artificery.Artificery(checkpoint_init=True)

    spec_path = os.path.expanduser(model_spec["spec_path"])
    spec_dir = os.path.dirname(spec_path)
    my_p = a.parse(model_spec["spec_path"])
    my_p.name = name

    if write_spec:
        model_spec_dst = os.path.join(checkpoint_folder, "model_spec.json")
        subprocess.Popen("ls {}".format(spec_path), shell=True)
        subprocess.Popen("cp {} {}".format(spec_path, model_spec_dst), shell=True)
        for sf in a.used_specfiles:
            sf = os.path.expanduser(sf)
            rel_folder = os.path.dirname(os.path.relpath(sf, spec_dir))
            dst_folder = os.path.join(checkpoint_folder, rel_folder)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
            subprocess.Popen("cp {} {}".format(sf, dst_folder), shell=True)

    if os.path.isfile(checkpoint_path):
        #my_p.load_state_dict(torch.load(checkpoint_path))
        state_dict = torch.load(checkpoint_path)

        own_state = my_p.state_dict()
        reinit_downmodules = [7]
        reinit_upmodules = []

        for layer_name, param in state_dict.items():
            load_weights = True
            if "level_downmodules" in layer_name:
                level = layer_name[len("level_downmodules.")]
                level = int(level[:1])
                if level in reinit_downmodules:
                    load_weights = False

            elif "level_upmodules" in layer_name:
                level = layer_name[len("level_upmodules.")]
                level = int(level[:1])
                if level in reinit_upmodules:
                    load_weights = False

            if layer_name not in own_state:
                load_weights = False

            if load_weights:
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                own_state[layer_name].copy_(param)
                #print ("Overwriting '{}'".format(layer_name))


    return my_p

def to_tensor(np_array, device='cpu'):
    return torch.tensor(np_array, device=device)

def expand_dims(tensor, dim_out):
    dim_in = len(list(tensor.shape))
    assert dim_out >= dim_in
    for i in range(dim_in, dim_out):
        tensor = tensor.unsqueeze(0)
    return tensor

def get_np(pt):
    if type(pt) == np.ndarray:
        return pt
    if pt.device == torch.device('cpu'):
        return pt.detach().numpy()
    return pt.cpu().detach().numpy()

def compose_functions(fseq):
    def compose(f1, f2):
        return lambda x: f2(f1(x))
    return reduce(compose, fseq, lambda _: _)


def np_upsample(img, factor):
    if factor == 1:
        return img

    if img.ndim == 2:
        return rescale(img, factor)
    elif img.ndim == 3:
        b = np.empty((int(img.shape[0] * factor),
                      int(img.shape[1] * factor), img.shape[2]))
        for idx in range(img.shape[2]):
            b[:, :, idx] = np_upsample(img[:, :, idx], factor)
        return b
    else:
        assert False


def np_downsample(img, factor):
    data_4d = np.expand_dims(img, axis=1)
    result = nn.AvgPool2d(factor)(torch.from_numpy(data_4d))
    return result.numpy()[:, 0, :, :]


def center_field(field):
    wrap = type(field) == np.ndarray
    if wrap:
        field = [field]
    for idx, vfield in enumerate(field):
        vfield[:, :, :, 0] = vfield[:,:,:,0] - np.mean(vfield[:,:,:,0])
        vfield[:, :, :, 1] = vfield[:,:,:,1] - np.mean(vfield[:,:,:,1])
        field[idx] = vfield
    return field[0] if wrap else field


def reverse_dim(var, dim):
    if var is None:
        return var
    idx = range(var.size()[dim] - 1, -1, -1)
    idx = torch.LongTensor(idx)
    if var.is_cuda:
        idx = idx.cuda()
    return var.index_select(dim, idx)


def reduce_seq(seq, f):
    size = min([x.size()[-1] for x in seq])
    return f([center(var, (-2, -1),
              var.size()[-1] - size) for var in seq], 1)


def center(var, dims, d):
    if not isinstance(d, collections.Sequence):
        d = [d for i in range(len(dims))]
    for idx, dim in enumerate(dims):
        if d[idx] == 0:
            continue
        var = var.narrow(dim, d[idx]/2, var.size()[dim] - d[idx])
    return var


def crop(data_2d, crop):
    return data_2d[..., crop:-crop, crop:-crop]


def downsample(x):
    if x > 0:
        return nn.AvgPool2d(2**x, count_include_pad=False)
    else:
        return (lambda y: y)


def upsample(x):
    if x > 0:
        return nn.Upsample(scale_factor=2**x, mode='bilinear')
    else:
        return (lambda y: y)

def normalize_bundle(bundle, per_feature_center=True,
        per_feature_var=True, mask_fill=None,
        mask_defects=True, mask_zeros=True):
    for name in ['src', 'tgt']:
        img = bundle[name]
        mask = torch.ones(img.shape[-2:],
                device=img.device, dtype=torch.uint8)
        if mask_zeros and (name + '_zeros') in bundle:
            zero_mask = bundle[name + '_zeros'].squeeze()
            mask[zero_mask != 0] = 0

        if mask_defects and (name + '_defects') in bundle:
            defect_mask = bundle[name + '_defects'].squeeze()
            mask[defect_mask != 0] = 0

        mask = mask.unsqueeze(0).unsqueeze(0).bool()
        bundle[name] = normalize(img, per_feature_center,
                        per_feature_var, mask=mask,
                        mask_fill=mask_fill)
    return bundle

def normalize(img, per_feature_center=True, per_feature_var=False, eps=1e-8,
        mask=None, mask_fill=None):
    img_out = img.clone()
    mask = mask.bool()
    #with torch.no_grad():
    if mask is not None: #black masks are shared accross featuremaps
        assert mask.shape[-1] == img.shape[-1]
    #if hasattr(mask, 'bool'):
    #    mask = mask.bool()
    mask = mask > 0
    for i in range(1):
        for b in range(img.shape[0]):
            x = img_out[b]

            if mask is not None:
                m = mask[b, 0]
            else:
                m = torch.ones_like(x[0])
            if per_feature_center and len(img.shape) == 4:
                for f in range(img.shape[1]):
                    x[f][m] = x[f][m].clone() - torch.mean(x[f][m].clone())
            else:
                x[:, m] = x[:, m].clone() - torch.mean(x[:, m].clone())
            if per_feature_var and len(img.shape) == 4:
                for f in range(img.shape[1]):
                    var = torch.var(x[f][m].clone())
                    if var == var:
                        x[f][m] = x[f][m].clone() / (torch.sqrt(var + eps))
            else:
                var = torch.var(x[:, m].clone())
                if var == var:
                    x[:, m] = x[:, m].clone() / (torch.sqrt(var + eps))

    if mask is not None and mask_fill is not None:
        for b in range(img.shape[0]):
            if len(img.shape) == 4:
                img_out[b, :, mask[b, 0].squeeze() == False] = mask_fill
            else:
                assert len(img.shape) == 3
                img_out[b, mask[b].squeeze() == False] = mask_fill

    return img_out

