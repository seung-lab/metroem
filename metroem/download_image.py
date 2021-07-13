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

from cloudvolume import CloudVolume
import numpy as np

from pathlib import Path
from tqdm import tqdm

def get_dset_path(dst_folder,
                  x_offset,
                  y_offset,
                  z_start,
                  mip,
                  suffix):
    """Get H5 filepath
    """
    suffix = '_' + suffix if suffix is not None else ''
    dset_name = "x{}_y{}_z{}_MIP{}{}.h5".format(x_offset,
            y_offset, z_start, mip, suffix)
    return dst_folder / dset_name

def write_array(dset, data, sample_index, pair_index):
    """Write ndarray to H5 file
    """
    dset[sample_index, pair_index] = data

def make_image_dset(dset_path,
                  num_samples,
                  patch_size,
                  chunk_size=512,
                  dtype=np.float32):
    """Define H5 file for image data

    Args:
        dset_path (str): H5 filepath
        num_samples (int)
        patch_size (int): W x H; W==H for each sample
        chunk_size (int): H5 chunking (default: patch_size)
        dtype (type): datatype of H5

    Returns:
        h5py.File object, sized:
            num_samples x 2 x patch_size x patch_size
    """
    print('make_field_dset')
    if chunk_size is None:
        chunk_size = patch_size
    data_name ='img'
    data_shape = [patch_size, patch_size]
    chunk_shape = [chunk_size, chunk_size]
    df = h5py.File(dset_path, 'a')
    dset_shape = (num_samples, 2, *data_shape)
    chunk_dim = (1, 1, *chunk_shape)
    if data_name in df:
        del df[data_name]
    dset = df.create_dataset(data_name,
                             dset_shape,
                             dtype=dtype,
                             chunks=chunk_dim,
                             compression='lzf',
                             scaleoffset=None)
    return dset

def download_section_image(vol,
                        dset,
                        x_offset,
                        y_offset,
                        z_range,
                        patch_size,
                        sample_index,
                        pair_index,
                        defect_mask,
                        mask_val):
    """Download image data with defects masked to H5 files

    Args:
        vol (CloudVolume)
        dset (h5py.File)
        x_offset (int)
        y_offset (int)
        z_range (slice): length one
        patch_size (int)
        sample_index (int)
        pair_index (int): (src, tgt): (0, 1)
        defect_mask (CloudVolume)
        mask_val (number): value of mask to exclude from image
    """
    img = vol[x_offset:x_offset + patch_size,
              y_offset:y_offset + patch_size,
              z_range].squeeze((2,3))
    print ((img != 0).sum())
    if defect_mask is not None:
        mask = defect_mask[x_offset:x_offset + patch_size,
                  y_offset:y_offset + patch_size,
                  z_range].squeeze((2,3))
        img[mask >= mask_val] = 0
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
                        offsets=None,
                        cv_path_defects=None,
                        mask_val=1):
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
        cv_path_defects (str): CloudVolume path to defect mask
        mask_val
    """
    section_ids = range(z_start, z_end)
    num_samples = len(section_ids)
    dset_path = get_dset_path(dst_folder=dst_folder,
                              x_offset=x_offset,
                              y_offset=y_offset,
                              z_start=z_start,
                              mip=mip,
                              suffix=suffix)
    dset = make_image_dset(dset_path=dset_path,
                          num_samples=num_samples,
                          patch_size=patch_size)
    vol = CloudVolume(cv_path,
                      mip=mip,
                      fill_missing=True,
                      bounded=False,
                      progress=False,
                      parallel=parallel)
    defect_mask = None
    if cv_path_defects is not None:
        defect_mask = CloudVolume(cv_path_defects,
                        mip=mip,
                        fill_missing=True,
                        bounded=False,
                        progress=False,
                        parallel=parallel)
    for sample_index, z in tqdm(enumerate(section_ids)):
        for pair_index in range(2):
            x_trans, y_trans = 0, 0
            if (offsets is not None):
                x_trans, y_trans = offsets[sample_index, pair_index, :]
            section_x_offset = (x_offset + x_trans) // 2**mip
            section_y_offset = (y_offset + y_trans) // 2**mip
            z_range = slice(z, z+1) if pair_index == 0 else slice(z-1, z)
            download_section_image(vol=vol,
                                dset=dset,
                                x_offset=section_x_offset,
                                y_offset=section_y_offset,
                                z_range=z_range,
                                patch_size=patch_size,
                                sample_index=sample_index,
                                pair_index=pair_index,
                                defect_mask=defect_mask,
                                mask_val=mask_val)

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
    parser.add_argument('--cv_path_defects', type=str, default=None)
    parser.add_argument('--mask_val', type=float, default=1)
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
                                offsets=offsets,
                                cv_path_defects=args.cv_path_defects,
                                mask_val=args.mask_val)
