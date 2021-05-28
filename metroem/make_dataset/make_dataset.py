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

from tqdm import tqdm
from pathlib import Path
from cloudvolume import CloudVolume
from cloudvolume.lib import Vec, Bbox


def spec_to_bbox(spec):
    return get_bbox(x_size=spec["x_size"], y_size=spec["y_size"], mip=spec["dst_mip"])


def get_bbox(x_size, y_size, mip):
    bbox_size = Vec(x_size, y_size, 1)
    bbox_size *= Vec(2 ** mip, 2 ** mip, 1)
    return Bbox((0, 0, 0), bbox_size)


def bbox_to_mip(bbox, from_mip, to_mip):
    """Adjust bbox to new mip

    Args:
        bbox (Bbox)
        from_mip (int)
        to_mip (int)

    Returns:
        Bbox
    """
    factor = 2 ** (to_mip - from_mip)
    scale = Vec(factor, factor, 1)
    dst_minpt = np.floor(bbox.minpt / scale).astype(int)
    dst_size = np.floor(bbox.size3() / scale).astype(int)
    return Bbox(dst_minpt, dst_minpt + dst_size)


class CloudTensor:
    def __init__(self, spec):
        self.vol = CloudVolume(
            spec["path"],
            mip=spec["src_mip"],
            fill_missing=True,
            bounded=False,
            progress=False,
            parallel=spec["parallel"],
        )

    def get(self, bbox, dst_mip):
        src_mip = self.vol.mip
        vol_bbox = bbox_to_mip(bbox, from_mip=0, to_mip=src_mip)
        im = self.vol[vol_bbox.to_slices()].squeeze((2, 3))
        im = im.astype(np.float32)
        im = torch.from_numpy(im).unsqueeze(dim=0).unsqueeze(dim=0)
        if src_mip != dst_mip:
            scale_factor = 2 ** (src_mip - dst_mip)
            im = torch.nn.functional.interpolate(
                im,
                mode="bilinear",
                scale_factor=scale_factor,
                recompute_scale_factor=False,
                align_corners=False,
            )
        return im

    @property
    def mip(self):
        return self.vol.mip

    def mip_resolution(self, mip):
        return self.vol.mip_resolution(mip)


class CloudField(CloudTensor):
    def get(self, bbox, dst_mip):
        src_mip = self.vol.mip
        assert src_mip >= dst_mip
        vol_bbox = bbox_to_mip(bbox, from_mip=0, to_mip=src_mip)
        field = self.vol[vol_bbox.to_slices()]
        field = np.transpose(field, (2, 3, 0, 1))
        field = torch.tensor(field).field()
        return field.up(mips=src_mip - dst_mip)


class Reference:
    """CloudTensor with fixed size MIP0 bbox"""

    def __init__(self, cloudtensor, bbox, dst_mip):
        self.vol = cloudtensor
        self.bbox = bbox
        self.dst_mip = dst_mip

    def get(self, bbox):
        return self.vol.get(self.adjust_bbox(bbox), dst_mip=self.dst_mip)

    def adjust_bbox(self, bbox):
        """Center reference bbox of fixed size at the center of input bbox."""
        ctr = bbox.minpt + (bbox.size() // Vec(2,2,1))
        return self.bbox + (ctr - self.bbox.size() // Vec(2,2,1))

    @property
    def src_mip(self):
        return self.vol.mip


class Mask(CloudTensor):
    def __init__(self, spec):
        super().__init__(spec)
        self.threshold = spec["threshold"]

    @classmethod
    def from_spec(cls, spec, **kwargs):
        spec.update(kwargs)
        return Mask(spec)

    def get(self, bbox, dst_mip):
        mask = super().get(bbox, dst_mip)
        return mask >= self.threshold


class Masks:
    """Collection of Mask objects"""

    def __init__(self, masks):
        self.masks = masks

    @classmethod
    def from_spec(cls, spec, **kwargs):
        masks = [Mask.from_spec(msp, **kwargs) for msp in spec]
        return Masks(masks)

    def get(self, bbox, dst_mip):
        vol_bbox = bbox_to_mip(bbox, from_mip=0, to_mip=dst_mip)
        x_size, y_size = vol_bbox.size3()[:2]
        cum_mask = torch.zeros((1, 1, x_size, y_size), dtype=bool)
        for mask in self.masks:
            ind_mask = mask.get(bbox, dst_mip)
            cum_mask = torch.logical_or(cum_mask, ind_mask)
        return cum_mask


class Image:
    """CloudTensor + Reference

    Reference is used to normalize image with consistent stats

    Args:
        vol (CloudTensor)
        masks (Masks)
        ref (Reference)
    """

    def __init__(self, vol, masks, ref):
        self.vol = vol
        self.masks = masks
        self.ref = ref

    @classmethod
    def from_spec(cls, spec, masks, **kwargs):
        """Create from spec json + Masks

        Args:
            spec (dict)
            masks (Masks)
        """
        spec.update(kwargs)
        im = CloudTensor(spec)
        spec["reference"].update(kwargs)
        ref = Reference(
            cloudtensor=CloudTensor(spec["reference"]),
            bbox=spec_to_bbox(spec["reference"]),
            dst_mip=spec["reference"]["dst_mip"],
        )
        return Image(im, masks=masks, ref=ref)

    @property
    def mip(self):
        return self.vol.mip

    def get(self, bbox, dst_mip, normalize=True):
        im = self.vol.get(bbox, dst_mip)
        im_black = im == 0
        if normalize:
            adj_bbox = self.ref.adjust_bbox(bbox)
            ref_im = im
            if self.ref.dst_mip != dst_mip:
                ref_im = self.ref.get(adj_bbox)
            ref_masks = self.masks.get(adj_bbox, self.ref.dst_mip)
            ref_im = ref_im[~ref_masks]
            ref_im = ref_im[ref_im != 0]
            ref_im = ref_im[ref_im == ref_im]
            mean = ref_im.mean()
            var = ref_im.var()
            if var == 0:
                print("Zero variance")
                var = torch.tensor(1)
            im = (im - mean) / var.sqrt()
        masks = self.masks.get(bbox, dst_mip)
        im[masks] = 0
        im[im_black] = 0
        return im


class Field:
    """CloudTensor + Reference

    Reference is used to profile field from consistent bbox

    Args:
        vol (CloudTensor)
        ref (Reference)
    """

    def __init__(self, vol, ref):
        self.vol = vol
        self.ref = ref

    @classmethod
    def from_spec(cls, spec, **kwargs):
        """Create from spec json

        Args:
            spec (dict)
        """
        spec.update(kwargs)
        im = CloudField(spec)
        spec["reference"].update(kwargs)
        ref = Reference(
            cloudtensor=CloudField(spec["reference"]),
            bbox=spec_to_bbox(spec["reference"]),
            dst_mip=spec["reference"]["dst_mip"],
        )
        return Field(im, ref=ref)

    @property
    def mip(self):
        return self.vol.mip

    def get(self, bbox, dst_mip):
        offset_field = self.ref.get(bbox)
        trans = offset_field.mean_finite_vector(keepdim=True)
        trans = (trans // (2 ** self.ref.src_mip)) * 2 ** self.ref.src_mip
        offset = trans.numpy()
        offset = Vec(int(trans[0, 1, 0, 0]), int(trans[0, 0, 0, 0]), 0)
        # adjust output field by translation
        field = self.vol.get(bbox, dst_mip)
        field -= trans
        field = field / (2 ** dst_mip)
        return field, offset


def download_sample(bbox, image, mip, normalize=True):
    """Download masked image pair to dst

    Args:
        bbox (Bbox): MIP0
        image (Image)
        mip (int)
        normalize (bool)
    """
    sample_image = image.get(bbox, mip, normalize=normalize)
    return sample_image


def download_sample_with_field(bbox, image, field, mip, normalize=True):
    """Download warped and masked image pair to dst

    Args:
        bbox (Bbox): MIP0
        image (Image)
        field (Field)
        mip (int)
        normalize (bool)
    """
    sample_field, offset = field.get(bbox, mip)
    adj_bbox = bbox + offset
    sample_image = image.get(adj_bbox, mip, normalize=normalize)
    return sample_field, sample_image


def write_to_cloudvolume(vol, data, sample_id):
    """Write image (warped if field present) to CloudVolume

    Args:
        vol (CloudVolume)
        data (dict): label, torch.Tensor
        sample_id (int)
    """
    im = data["image"]
    if "field" in data.keys():
        f = data["field"].from_pixels()
        im = f(im)
    im = im.permute((2, 3, 0, 1))
    # im = torch.round(im).to(dtype=torch.uint8)
    if im.is_cuda:
        im = im.cpu()
    im = im.numpy()
    if vol.data_type == "uint8":
        im = im.astype(np.uint8)
    dst_bbox = Bbox(Vec(0, 0, sample_id), vol.chunk_size + Vec(0, 0, sample_id))
    vol[dst_bbox.to_slices()] = im


def write_to_h5(h5_dset, data, sample_id):
    """Write image & field to respective H5

    Args:
        h5_dset (dict): label, H5 file
        data (dict): label, torch.Tensor
        sample_id (int)
    """
    for k, obj in data.items():
        sample_index = sample_id // 2
        pair_index = sample_id % 2
        h5f = h5_dset[k]
        obj = data[k]
        if obj.is_cuda:
            obj = obj.cpu()
        obj = obj.numpy()
        if k == "image":
            h5f[sample_index, pair_index] = obj
        elif k == "field":
            h5f[sample_index] = obj


def get_dst_cloudvolume(spec, dst_path, res, parallel=1, data_type="float32"):
    """Create CloudVolume where result will be written

    Args:
        spec (dict)
        dst_path (str)
        res (Vec): MIP0
        parallel (int)

    Returns:
        CloudVolume
    """
    mip = spec["dst_mip"]
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="image",
        data_type=data_type,
        encoding="raw",
        resolution=res,
        voxel_offset=[0, 0, 0],
        chunk_size=[spec["x_size"], spec["y_size"], 1],
        volume_size=[
            spec["x_size"] * 2 ** mip,
            spec["y_size"] * 2 ** mip,
            2 * len(spec["pairs"]),
        ],
    )

    dst = CloudVolume(dst_path, mip=0, info=info, cdn_cache=False, parallel=parallel)
    for m in range(1, mip + 1):
        factor = Vec(2 ** m, 2 ** m, 1)
        dst.add_scale(factor=factor, chunk_size=[spec["x_size"], spec["y_size"], 1])

    dst.commit_info()
    dst = CloudVolume(dst_path, mip=mip, cdn_cache=False, parallel=parallel)
    return dst


def get_h5_path(spec, dst_path, prefix, suffix):
    """Get H5 filepath"""
    prefix = prefix + "_" if prefix is not None else ""
    suffix = "_" + suffix if suffix is not None else ""
    sample = spec["pairs"][0][0]  # first sample
    dset_name = "{}x{}_y{}_z{}_MIP{}{}.h5".format(
        prefix,
        sample["x"],
        sample["y"],
        sample["z"],
        spec["dst_mip"],
        suffix,
    )
    return dst_path / dset_name


def get_h5_dset(spec, dst_path):
    """Create set of H5 files to store image & field

    Args:
        spec (dict)

    Returns:
        dict, {'image', 'field'} : H5 file
    """
    n = len(spec["pairs"])
    image_path = get_h5_path(spec, dst_path, prefix=None, suffix=None)
    field_path = get_h5_path(spec, dst_path, prefix="field_0", suffix="test")
    params = {
        "image": {
            "name": "img",
            "path": image_path,
            "shape": (n, 2, spec["x_size"], spec["y_size"]),
            "chunks": (1, 1, spec["x_size"], spec["y_size"]),
            "scaleoffset": None,
        },
        "field": {
            "name": "field",
            "path": field_path,
            "shape": (n, 2, spec["x_size"], spec["y_size"]),
            "chunks": (1, 2, spec["x_size"], spec["y_size"]),
            "scaleoffset": 2,
        },
    }
    dsets = {}
    for name, param in params.items():
        df = h5py.File(param["path"], "w")
        dset = df.create_dataset(
            "main",
            shape=param["shape"],
            dtype=np.float32,
            chunks=param["chunks"],
            compression="lzf",
            scaleoffset=param["scaleoffset"],
        )
        dsets[name] = dset
    return dsets


def get_chunk_size(dst, to_cloudvolume=False):
    if to_cloudvolume:
        return dst.chunk_size
    else:
        dst_shape = dst["image"].shape
        return Vec(dst_shape[-2], dst_shape[-1], 1)


def make_dataset(spec, dst_path, to_cloudvolume=False, normalize=True):
    """Create CloudVolume containing MetroEM dataset

    Dataset will contain warped & masked src,tgt image pairs

    Args:
        spec (dict): specification object; see README.md
        dst_path (str): CloudVolume path
        to_cloudvolume (bool)
        normalize (bool)
    """
    parallel = spec.get("parallel", 1)
    masks = Masks.from_spec(spec["masks"], parallel=parallel)
    field = None
    if "field" in spec.keys():
        field = Field.from_spec(spec["field"], parallel=parallel)
    image = Image.from_spec(spec["image"], masks=masks, parallel=parallel)
    bbox = spec_to_bbox(spec)
    if to_cloudvolume:
        data_type = "float32" if normalize else "uint8"
        dst = get_dst_cloudvolume(
            spec=spec,
            dst_path=dst_path,
            res=image.vol.mip_resolution(0),
            parallel=parallel,
            data_type=data_type,
        )
    else:
        dst = get_h5_dset(spec=spec, dst_path=dst_path)

    for k, pair in enumerate(spec["pairs"]):
        for i, sample_spec in enumerate(pair):
            center = Vec(*[sample_spec[v] for v in ["x", "y", "z"]])
            size = Vec(spec["x_size"], spec["y_size"], 0)
            size = size * 2 ** spec["dst_mip"] // 2
            offset = center - size
            if field is None:
                im = download_sample(
                    bbox=bbox + offset,
                    image=image,
                    mip=spec["dst_mip"],
                    normalize=normalize,
                )
                data = {"image": im}
            else:
                f, im = download_sample_with_field(
                    bbox=bbox + offset,
                    image=image,
                    field=field,
                    mip=spec["dst_mip"],
                    normalize=normalize,
                )
                data = {"field": f, "image": im}
                if not to_cloudvolume:
                    if i % 2 == 1:
                        del data["field"]
                        im = f.from_pixels()(im)
                        data["image"] = im

            sample_id = 2 * k + i
            print("Writing sample {}".format(sample_id))
            if to_cloudvolume:
                write_to_cloudvolume(dst, data=data, sample_id=sample_id)
            else:
                write_to_h5(dst, data=data, sample_id=sample_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetroEM dataset from CloudVolume")
    parser.add_argument("--spec_path", type=str, help="See create_spec.py")
    parser.add_argument(
        "--dst_path",
        type=str,
        help="Use local path to create H5 file; \
                    use cloudpath to create new CloudVolume",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--unnormalize", action="store_true")

    args = parser.parse_args()
    with open(args.spec_path, "r") as f:
        spec = json.load(f)

    dst_path = Path(args.dst_path)
    to_cloudvolume = dst_path.parts[0] == "gs:"
    if not to_cloudvolume:
        dst_path.mkdir(parents=True, exist_ok=True)
    else:
        dst_path = args.dst_path
    spec["parallel"] = args.parallel

    make_dataset(
        spec=spec,
        dst_path=dst_path,
        to_cloudvolume=to_cloudvolume,
        normalize=not args.unnormalize,
    )
