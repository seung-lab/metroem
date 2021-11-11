import sys
import json
import os
import time
import six
import argparse
import glob
import re
import h5py

import modelhouse
import numpy as np
import torch
import torchfields

from metroem import helpers

from cloudvolume import CloudVolume
from cloudvolume.lib import Vec

from pdb import set_trace as st

def get_pyramid_modules(pyramid_path):
    module_dict = {}
    contents = next(os.walk(pyramid_path))[1]

    for m in contents:
        pattern = r"(\d)_mip(\d)in_(.*)"
        match = re.search(pattern, m)
        if match is None:
            raise Exception(f"Invalid member in the pyramid: {m}.\n"\
                    "Pyramid folder must only contain model folders with names in " \
                    "format '{0-9}_mip{0-9}in_{arbitrary name}'")
        module_id = int(match.group(1))
        module_mip_in = int(match.group(2))
        module_path = os.path.join(pyramid_path, m)
        module_dict[module_id] = {"path": module_path, "mip_in": module_mip_in}

    return module_dict

def generate_fields(pyramid_path,
                 stage,
                 checkpoint_name,
                 img_path,
                 prev_field_path,
                 dst_dir,
                 src_mip,
                 dst_mip,
                 device="cuda",
                 restart_index=0):
    """Generate field for image pairs

    Args:
        pyramid_path (str): path to pyramid directory
        stage (int): module index within the pyramid
        checkpoint_name (str): checkpoint for weights
        img_path (str): path to image pairs h5
        prev_field_path (str): path to previous fields h5
        dst_dir (str): path where temporary field h5s will be stored
        src_mip (int)
        dst_mip (int)
        device (str)
        restart_index (int)
    """
    pyramid_path = os.path.expanduser(pyramid_path)
    module_dict = get_pyramid_modules(pyramid_path)
    module_path = module_dict[stage]["path"]
    model = modelhouse.load_model_simple(module_path,
                                         finetune=True,
                                         finetune_lr=0.15,
                                         finetune_sm=600, 
                                         finetune_iter=600,
                                         pass_field=True,
                                         checkpoint_name=checkpoint_name)
    checkpoint_path = os.path.join(module_path, "model")

    img_dset = h5py.File(img_path, 'r')['main']
    prev_field_dset = h5py.File(prev_field_path, 'r')['main']
    n_start = 0
    n_stop = img_dset.shape[0]
    src_mip_filepath = os.path.join(dst_dir, 
                                '{}'.format(src_mip))
    dst_mip_filepath = os.path.join(dst_dir, 
                                '{}'.format(dst_mip))
    src_field_dset = CloudVolume(src_mip_filepath, mip=0)
    dst_field_dset = CloudVolume(dst_mip_filepath, mip=0)
    # src_field_dset = src_field.create_dataset("main", 
    #                                        shape=field_shape,
    #                                        dtype=np.float32,
    #                                        chunks=chunks,
    #                                        compression='lzf',
    #                                        scaleoffset=2)
    # dst_field_dset = dst_field.create_dataset("main", 
    #                                        shape=field_shape,
    #                                        dtype=np.float32,
    #                                        chunks=chunks,
    #                                        compression='lzf',
    #                                        scaleoffset=2)

    for b in range(restart_index, n_stop):
        print('{} / {}'.format(img_dset.shape[0], b))
        src = helpers.to_tensor(img_dset[b, 0], device=device)
        tgt = helpers.to_tensor(img_dset[b, 1], device=device)
        if prev_field_dset is not None:
            prev_field = helpers.to_tensor(prev_field_dset[b], device=device)
        else:
            prev_field = None

        field = model(src_img=src, 
                      tgt_img=tgt, 
                      src_agg_field=prev_field, 
                      train=False,
                      return_state=False)
        field_shape = field.shape
        hsz = (src_field_dset.shape[0] * 2**(src_mip-dst_mip) - dst_field_dset.shape[0]) // 2
        src_field_dset[:, :, b - n_start, :] = helpers.get_np(field.permute(2,3,0,1))
        # upsample
        field = field * (2**src_mip)
        field = field.up(mips=src_mip - dst_mip)
        field = field / (2**dst_mip)
        field_cropped = field[:, :, hsz:-hsz, hsz:-hsz]
        field_cropped = field_cropped.permute(2,3,0,1)
        dst_field_dset[:, :, b - n_start, :] = helpers.get_np(field_cropped)


def main():
    parser = argparse.ArgumentParser(description='Generate field for training set module.')

    parser.add_argument('--pyramid_path', type=str)
    parser.add_argument('--checkpoint_name', type=str, default="checkpoint")
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--prev_field_path', type=str)
    # parser.add_argument('--field_path', type=str)
    parser.add_argument('--dst_dir', type=str)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--src_mip', type=int, help='MIP of input')
    parser.add_argument('--dst_mip', type=int, help='MIP of output')
    parser.add_argument('--restart_index', type=int, help='Sample ID for restart', default=0)

    parser.set_defaults(redirect_stdout=True)
    args = parser.parse_args()

    assert(os.path.exists(args.pyramid_path))
    assert(os.path.exists(args.image_path))
    assert(os.path.exists(args.prev_field_path))
    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

    img_dset = h5py.File(args.image_path, 'r')['main']
    down_res = Vec(2**args.src_mip, 2**args.src_mip, 1)
    up_res = Vec(2**args.dst_mip, 2**args.dst_mip, 1)
    for mip in [args.src_mip, args.dst_mip]:
        res = Vec(2**mip, 2**mip, 1)
        path = os.path.join(args.dst_dir, str(mip))
        info = CloudVolume.create_new_info(
                        num_channels = 2,
                        layer_type = 'image',
                        data_type = 'float32',
                        encoding = 'raw',
                        resolution = res,
                        voxel_offset = [0, 0, 0],
                        chunk_size = [img_dset.shape[-1],
                                      img_dset.shape[-2],
                                      1],
                        volume_size = [img_dset.shape[-1],
                                       img_dset.shape[-2],
                                       img_dset.shape[0]])
        cv = CloudVolume(path, 
                        mip=0,
                        info=info,
                        cdn_cache=False)
        cv.commit_info()

    generate_fields(pyramid_path=args.pyramid_path,
                  stage=args.stage,
                  checkpoint_name=args.checkpoint_name,
                  img_path=args.image_path,
                  prev_field_path=args.prev_field_path,
                  dst_dir=args.dst_dir,
                  src_mip=args.src_mip,
                  dst_mip=args.dst_mip,
                  device="cpu" if args.gpu is None else "cuda",
                  restart_index=args.restart_index)


if __name__ == "__main__":
    main()
