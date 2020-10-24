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
from helpers import write_tensor, \
                    make_dset, \
                    get_dset_path, \
                    get_cv_and_dset

def make_offset_dset(dset_path, 
                    num_samples, 
                    dtype=int):
    """Define H5 file for field offsets

    Args:
        dset_path (str): H5 filepath
        num_samples (int)
        dtype (type): datatype of H5 

    Returns:
        h5py.Dataset object, sized: 
            num_samples x 2 (src, tgt) x 2 (x, y)
    """
    df = h5py.File(dset_path, 'a')
    data_name = 'offset'
    if data_name in df:
        del df[data_name]
    dset_shape = (num_samples, 2, 2)
    chunk_dim = (1, 1, 2)
    return df.create_dataset(data_name,
                             dset_shape, 
                             dtype=int,
                             chunks=chunk_dim,
                             compression='lzf')

def download_section_field(vol,    
                        dset,
                        offsets,
                        x_offset,
                        y_offset,
                        z,
                        mip,
                        patch_size,
                        sample_index,
                        pair_index):
    """Download field to H5 file and collect offset adjustments

    The field will not be used to warp the img and defects. Warping is handled 
    in the dataloader.

    Field is assumed to be in MIP0 displacements. Translations are stored in
    MIP0 displacements.

    Args:
        vol (CloudVolume)
        dset (h5py.Dataset)
        offsets (h5py.Dataset): z x src/tgt x x/y translation
        x_offset (int)
        y_offset (int)
        z (int)
        mip (int)
        patch_size (int)
        sample_index (int)
        pair_index (int): (src, tgt): (0, 1)
    """
    z_range = slice(z, z+1) if pair_index == 0 else slice(z-1, z)
    field = vol[x_offset:x_offset + patch_size,
                y_offset:y_offset + patch_size,
                z_range]
    field = np.transpose(field, (2,3,0,1))
    field = torch.tensor(field).field_()
    trans = field.mean_finite_vector(keepdim=True)
    trans = (trans // (2**mip)) * 2**mip
    offsets[sample_index, pair_index, :] = [int(trans[0,0,0,0]), int(trans[0,1,0,0])]
    field -= trans
    field = field.permute(0,2,3,1).squeeze()
    write_tensor(dset=dset,
                 data=field,
                 sample_index=sample_index,
                 pair_index=pair_index)


def download_dataset_field(cv_path,
                        dst_folder, 
                        z_start, 
                        z_end,
                        mip, 
                        x_offset=0, 
                        y_offset=0, 
                        patch_size=None,
                        suffix=None,
                        parallel=1):
    """Create CloudVolume & H5 file and transfer field

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
    """
    section_ids = range(z_start, z_end)
    num_samples = len(section_ids)
    dset_path = get_dset_path(dst_folder=dst_folder,
                              x_offset=x_offset,
                              y_offset=y_offset,
                              z_start=z_start,
                              mip=mip,
                              suffix=suffix)
    vol, dset = get_cv_and_dset(data_name='field',
                                cv_path=cv_path,
                                dset_path=dset_path,
                                num_samples=num_samples,
                                mip=mip, 
                                patch_size=patch_size,
                                suffix=suffix,
                                parallel=parallel)
    offsets = make_offset_dset(dset_path=dset_path, 
                               num_samples=num_samples, 
                               dtype=int)
    x_offset //= 2**mip
    y_offset //= 2**mip
    for sample_index, z in tqdm(enumerate(section_ids)):
        for pair_index in range(2):
            download_section_field(vol=vol,
                                dset=dset,
                                offsets=offsets,
                                x_offset=x_offset,
                                y_offset=y_offset,
                                z=z,
                                mip=mip,
                                patch_size=patch_size,
                                sample_index=sample_index,
                                pair_index=pair_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Create MetroEM field dataset via CloudVolume')
    parser.add_argument('--mip',         type=int)
    parser.add_argument('--patch_size',  type=int)
    parser.add_argument('--x_offset',    type=int, default=0)
    parser.add_argument('--y_offset',    type=int, default=0)
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
    parser.add_argument('--suffix', type=str, default=None)
    parser.add_argument('--dst_folder', type=str, default='./dataset01')
    parser.add_argument('--parallel', type=int, default=1)

    args = parser.parse_args()

    dst_folder = Path(args.dst_folder)
    dst_folder.mkdir(parents=True, exist_ok=True)

    assert(args.x_offset % 2**args.mip == 0)
    assert(args.y_offset % 2**args.mip == 0)

    download_dataset_field(cv_path=args.cv_path,
                        dst_folder=dst_folder,
                        z_start=args.z_start, 
                        z_end=args.z_end,
                        mip=args.mip, 
                        x_offset=args.x_offset, 
                        y_offset=args.y_offset, 
                        patch_size=args.patch_size,
                        suffix=args.suffix,
                        parallel=args.parallel)
