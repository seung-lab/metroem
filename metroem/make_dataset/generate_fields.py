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

from metroem import helpers

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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

def setup(rank, world_size):
    dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank)

def cleanup():
    dist.destroy_process_group()

def generate_shard(rank, 
                 world_size,
                 module_path,
                 checkpoint_name,
                 img_path,
                 prev_field_path,
                 temp_dir):
    """Generate field for subset of image pairs associated with rank

    Args:
        rank (int): process order
        world_size (int): total no. of processes
        module_path (str): path to modelhouse directory
        checkpoint_name (str): checkpoint for weights
        img_path (str): path to image pairs h5
        prev_field_path (str): path to previous fields h5
        temp_dir (str): path where temporary field h5s will be stored
    """
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    model = modelhouse.load_model_simple(module_path,
                                         finetune=True,
                                         finetune_lr=3e-1, 
                                         finetune_sm=60e0, 
                                         finetune_iter=200,
                                         pass_field=True,
                                         checkpoint_name=checkpoint_name)
    checkpoint_path = os.path.join(module_path, "model")
    model.aligner.net = model.aligner.net.to(rank)
    model = DDP(model, device_ids=[rank])

    img_dset = h5py.File(img_path, 'r')['main']
    prev_field_dset = h5py.File(prev_field_path, 'r')['main']
    n = img_dset.shape[0] // world_size
    n_start = rank * n
    n_stop = min(n_start + n, img_dset.shape[0])
    tmp_filepath = os.path.join(temp_dir, '{}_{}.h5'.format(n_start, n_stop))
    field = h5py.File(tmp_filepath, 'w')
    field_shape = (n, 2, img_dset.shape[-2], img_dset.shape[-1])
    chunks = (1, 2, img_dset.shape[-2], img_dset.shape[-1])
    field_dset = field.create_dataset("main", 
                                           shape=field_shape,
                                           dtype=np.float32,
                                           chunks=chunks,
                                           compression='lzf',
                                           scaleoffset=2
                                           )

    for b in range(n_start, n_stop):
        print('{} / {}'.format(img_dset.shape[0], b))
        src = helpers.to_tensor(img_dset[b, 0])
        tgt = helpers.to_tensor(img_dset[b, 1])
        if prev_field_dset is not None:
            prev_field = helpers.to_tensor(prev_field_dset[b])
        else:
            prev_field = None

        field = model(src_img=src, 
                      tgt_img=tgt, 
                      src_agg_field=prev_field, 
                      train=False,
                      return_state=False)
        field_dset[b-n_start] = helpers.get_np(field)

    cleanup()
    pass

def generate_shards_distributed(world_size,
                  pyramid_path, 
                  stage, 
                  checkpoint_name,
                  img_path,
                  prev_field_path,
                  temp_dir):
    pyramid_path = os.path.expanduser(pyramid_path)
    module_dict = get_pyramid_modules(pyramid_path)
    module_path = module_dict[stage]["path"]

    mp.spawn(generate_shard,
             args=(world_size,
                   module_path,
                   checkpoint_name,
                   img_path,
                   prev_field_path,
                   temp_dir),
             nprocs=world_size,
             join=True)


def main():
    parser = argparse.ArgumentParser(description='Generate field for training set module.')

    parser.add_argument('--pyramid_path', type=str)
    parser.add_argument('--checkpoint_name', type=str, default="checkpoint")
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--prev_field_path', type=str)
    # parser.add_argument('--field_path', type=str)
    parser.add_argument('--temp_dir', type=str)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--stage', type=int)
    parser.add_argument('--port', type=int, default=8888)

    parser.set_defaults(redirect_stdout=True)
    args = parser.parse_args()

    assert(os.path.exists(args.pyramid_path))
    assert(os.path.exists(args.image_path))
    assert(os.path.exists(args.prev_field_path))
    os.mkdir(args.temp_dir)

    world_size = len(args.gpu.split(','))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)

    generate_shards_distributed(world_size=world_size,
                  pyramid_path=args.pyramid_path,
                  stage=args.stage,
                  checkpoint_name=args.checkpoint_name,
                  img_path=args.image_path,
                  prev_field_path=args.prev_field_path,
                  temp_dir=args.temp_dir)
    # gather_shards(temp_dir=args.temp_dir,
    #               dst_path=args.field_path)


if __name__ == "__main__":
    main()
