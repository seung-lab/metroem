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

import numpy as np

from pathlib import Path
from tqdm import tqdm
from cloudvolume import CloudVolume

def get_dset_path(dst_folder,
                  x_offset,
                  y_offset,
                  z_start,
                  mip,
                  suffix):
    """Get H5 filepath
    """
    suffix = '_' + suffix if suffix is not None else ''
    dset_name = "field_0_x{}_y{}_z{}_MIP{}{}.h5".format(x_offset,
                                                        y_offset,
                                                        z_start,
                                                        mip,
                                                        suffix)
    return dst_folder / dset_name

def write_tensor(dset, data, sample_index, pair_index):
    """Write tensor to H5 file

    Args:
        dset (h5py.File)
        data (torch.Tensor): no leading identity dimensions
        sample_index (int)
        pair_index (int)
    """
    if data.is_cuda:
        data = data.cpu()
    data = data.numpy()
    dset[sample_index, pair_index] = data

def make_field_dset(dset_path,
                  num_samples,
                  patch_size,
                  chunk_size=512,
                  dtype=np.float32):
    """Define H5 file for data_name

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
    data_name = 'field'
    scaleoffset = 2
    df = h5py.File(dset_path, 'a')
    dset_shape = (num_samples, 2, 2, patch_size, patch_size)
    chunk_dim = (1, 1, 2, chunk_size, chunk_size)
    if data_name in df:
        del df[data_name]
    dset = df.create_dataset(data_name,
                             dset_shape,
                             dtype=dtype,
                             chunks=chunk_dim,
                             compression='lzf',
                             scaleoffset=scaleoffset)
    return dset

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
                        z_range,
                        dst_mip,
                        patch_size,
                        sample_index,
                        pair_index):
    """Download field to H5 file and collect offset adjustments

    The field will not be used to warp the img and defects. Warping is handled
    in the dataloader.

    Field is assumed to be in MIP0 displacements, and converted to
    MIP displacements. Translations are stored in MIP0 displacements.

    Args:
        vol (CloudVolume)
        dset (h5py.Dataset)
        offsets (h5py.Dataset): z x src/tgt x x/y translation
        x_offset (int): vol.mip pixels
        y_offset (int): vol.mip pixels
        z_range (slice): length one
        dst_mip (int): MIP of dataset
        patch_size (int): dst_mip pixels
        sample_index (int)
        pair_index (int): (src, tgt): (0, 1)
    """
    src_mip = vol.mip
    assert(src_mip > dst_mip)
    scale_factor = 2**(src_mip - dst_mip)
    in_field = vol[x_offset:x_offset + (patch_size // scale_factor),
                   y_offset:y_offset + (patch_size // scale_factor),
                   z_range]
    in_field = np.transpose(in_field, (2,3,0,1))
    in_field = torch.tensor(in_field).field()
    in_field = in_field.squeeze()
    out_field = in_field.up(src_mip - dst_mip)
    out_field = out_field[:, :patch_size, :patch_size]
    trans = out_field.mean_finite_vector(keepdim=True)
    trans = (trans // (2**dst_mip)) * 2**dst_mip
    offset = [int(trans[0,0,0]), int(trans[1,0,0])]
    offsets[sample_index, pair_index, :] = offset
    out_field -= trans
    out_field = out_field / (2**dst_mip)
    out_field = torch.flip(out_field, [0]) # reverse x,y components
    write_tensor(dset=dset,
                 data=out_field,
                 sample_index=sample_index,
                 pair_index=pair_index)

def download_dataset_field(cv_path,
                        dst_folder,
                        z_start,
                        z_end,
                        src_mip,
                        dst_mip,
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
        src_mip (int): MIP level of CloudVolume field
        dst_mip (int): MIP level of output dataset
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
                              mip=dst_mip,
                              suffix=suffix)
    dset = make_field_dset(dset_path=dset_path,
                          num_samples=num_samples,
                          patch_size=patch_size)
    vol = CloudVolume(cv_path,
                      mip=src_mip,
                      fill_missing=True,
                      bounded=False,
                      progress=False,
                      parallel=parallel)
    offsets = make_offset_dset(dset_path=dset_path,
                               num_samples=num_samples,
                               dtype=int)
    x_offset //= 2**src_mip
    y_offset //= 2**src_mip
    for sample_index, z in tqdm(enumerate(section_ids)):
        for pair_index in range(2):
            z_range = slice(z, z+1) if pair_index == 0 else slice(z-1, z)
            download_section_field(vol=vol,
                                dset=dset,
                                offsets=offsets,
                                x_offset=x_offset,
                                y_offset=y_offset,
                                z_range=z_range,
                                dst_mip=dst_mip,
                                patch_size=patch_size,
                                sample_index=sample_index,
                                pair_index=pair_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Create MetroEM field dataset via CloudVolume')
    parser.add_argument('--src_mip',         type=int)
    parser.add_argument('--dst_mip',         type=int)
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

    assert(args.x_offset % 2**args.src_mip == 0)
    assert(args.y_offset % 2**args.src_mip == 0)

    download_dataset_field(cv_path=args.cv_path,
                        dst_folder=dst_folder,
                        z_start=args.z_start,
                        z_end=args.z_end,
                        src_mip=args.src_mip,
                        dst_mip=args.dst_mip,
                        x_offset=args.x_offset,
                        y_offset=args.y_offset,
                        patch_size=args.patch_size,
                        suffix=args.suffix,
                        parallel=args.parallel)
