import time
import h5py
import json
import sys
import os
import six
import copy
import argparse

import torch
import torchfields

import cloudvolume as cv
import numpy as np

from pathlib import Path
from tqdm import tqdm
from helpers import write_array, \
                    make_dset, \
                    get_dset_path, \
                    get_cv_and_dset

def download_section_image(vol,
                        dset,
                        x_offset,
                        y_offset,
                        z,
                        patch_size,
                        sample_index,
                        pair_index):
    """Download image & defects data to H5 files

    Args:
        vol (CloudVolume)
        dset (h5py.File)
        x_offset (int)
        y_offset (int)
        z (int)
        patch_size (int)
        sample_index (int)
        pair_index (int): (src, tgt): (0, 1)
    """
    z_range = slice(z, z+1) if pair_index == 0 else slice(z-1, z)
    img = vol[x_offset:x_offset + patch_size,
              y_offset:y_offset + patch_size,
              z_range].squeeze((2,3))
    write_array(dset=dset,
                data=img,
                sample_index=sample_index,
                pair_index=pair_index)

def download_dataset_image(cv_path,
                        dst_folder, 
                        z_start, 
                        z_end,
                        mip, 
                        x_offset=0, 
                        y_offset=0, 
                        patch_size=None,
                        suffix=None,
                        parallel=1,
                        offsets=None):
    """Create CloudVolume & H5 file and transfer image

    Args:
        cv_path (str): CloudVolume path
        dst_folder (str): root of directory where H5 files will be stored
        offset_translations (dict): z: (x trans, y trans)
        z_start (int)
        z_end (int)
        mip (int)
        x_offset (int): offset in MIP0 pixels; must be multiple of MIP factor
        y_offset (int): offset in MIP0 pixels; must be multiple of MIP factor
        patch_size (int): width & height of 2D region to download
        suffix (str): append to each H5 filename
        parallel (int): no. of threads for CloudVolume operations
        offsets (h5py.Dataset): N x 2 x 2 dataset
    """
    section_ids = range(z_start, z_end)
    num_samples = len(section_ids)
    dset_path = get_dset_path(dst_folder=dst_folder,
                              x_offset=x_offset,
                              y_offset=y_offset,
                              z_start=z_start,
                              mip=mip,
                              suffix=suffix)
    vol, dset = get_cv_and_dset(data_name='img',
                                cv_path=cv_path,
                                dset_path=dset_path,
                                num_samples=num_samples,
                                mip=mip, 
                                patch_size=patch_size,
                                suffix=suffix,
                                parallel=1)
    for sample_index, z in tqdm(enumerate(section_ids)):
        for pair_index in range(2):
            x_trans, y_trans = 0, 0
            if offsets is not None:
                x_trans, y_trans = offsets[sample_index, pair_index,:]
            section_x_offset = (x_offset + x_trans) // 2**mip
            section_y_offset = (y_offset + y_trans) // 2**mip
            download_section_image(vol=vol,
                                dset=dset,
                                x_offset=section_x_offset,
                                y_offset=section_y_offset,
                                z=z,
                                patch_size=patch_size,
                                sample_index=sample_index,
                                pair_index=pair_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Create MetroEM image dataset via CloudVolume')
    parser.add_argument('--mips',        type=int, nargs='+')
    parser.add_argument('--patch_sizes', type=int, nargs='+')
    parser.add_argument('--x_offset',  type=int, default=0)
    parser.add_argument('--y_offset',  type=int, default=0)
    parser.add_argument(
            '--z_start', 
            type=int, 
            default=None,
            help='Start of source image range (target range start is z_start-1')
    parser.add_argument(
            '--z_end', 
            type=int, 
            default=None,
            help='End of source image range (target range end is z_end-1')
    parser.add_argument('--cv_path', type=str, default=None)
    parser.add_argument('--field_dset', 
                        type=str, 
                        default=None,
                        help='Path to field dset with offsets')
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--dst_folder', type=str, default='./dataset01')
    parser.add_argument('--parallel', type=int, default=1)

    args = parser.parse_args()
    assert(len(args.mips) == len(args.patch_sizes))
    dst_folder = Path(args.dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    offsets = None
    if args.field_dset is not None:
        offsets = h5py.File(args.field_dset, 'r')['offset']

    for mip, patch_size in zip(args.mips, args.patch_sizes):
        download_dataset_image(cv_path=args.cv_path,
                                dst_folder=dst_folder, 
                                z_start=args.z_start, 
                                z_end=args.z_end,
                                mip=mip, 
                                x_offset=args.x_offset, 
                                y_offset=args.y_offset, 
                                patch_size=patch_size,
                                suffix=args.suffix,
                                parallel=args.parallel,
                                offsets=offsets)
