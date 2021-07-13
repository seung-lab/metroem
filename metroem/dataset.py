from pdb import set_trace as st
import glob
import h5py
import os
import re
import torch
import torchfields
import numpy as np

from collections import defaultdict

from metroem.alignment import align_sample
from metroem import helpers


class MyConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        self.size_limit = super().__len__()

    def __len__(self):
        return self.size_limit

    def set_size_limit(self, size_limit):
        self.size_limit = min(size_limit, super().__len__())


class SimpleH5Dataset(torch.utils.data.Dataset):
    def __init__(self, dset):
        self.dset = dset

    def __len__(self):
        return dset.shape[0]

    def __getitem__(self, i):
        return self.dset[i]


class MultimipDataset:
    """Access image & field samples by MIP for dataset directory

    Filename conventions within dataset directory:
      Image: '<NAME>_MIP>MIP.H5'
      Field: 'field_<STAGE>_<NAME>_<MIP>MIP_<CHECKPOINT NAME>.H5'
        <NAME>: x<X_START>_y<Y_START>_z<Z_START>

    Fields are associated with a checkpoint name, because different experiments
    may produce different fields as intermediaries between modules.
    """

    def __init__(self, path, aug_params=None, field_tag=None, device="cuda"):
        self.path = path
        self.aug_params = aug_params
        self.data_piece_registry = {}
        self.min_mip = 9
        self.max_mip = 0
        self.names = set()
        self.device = device

        if field_tag is None:
            self.field_tag = ""
        else:
            self.field_tag = "_" + field_tag

        self.img_dsets = defaultdict(lambda: defaultdict(lambda: None))
        self.field_dsets = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: None))
        )
        self.img_composite_dsets = dict()
        self.field_composite_dsets = defaultdict(lambda: defaultdict(lambda: None))
        self.load()

    def load(self):
        pattern = r"x(\d+)_y(\d+)_z(\d+)_(.*)MIP(\d).h5"
        files = glob.glob(os.path.join(self.path, "*.h5"))

        for f_abs in files:
            f = os.path.basename(f_abs)
            match = re.search(pattern, f)
            if match is not None:
                name = "x{}_y{}_z{}{}".format(
                    match.group(1), match.group(2), match.group(3), match.group(4)
                )
                mip = int(match.group(5))
                self.max_mip = max(mip, self.max_mip)
                self.min_mip = min(mip, self.min_mip)

                if name not in self.names:
                    self.names.add(name)
                    print(f"Adding '{name}' dataset.")

        for n in self.names:

            for mip in range(self.min_mip, self.max_mip + 1):
                self.load_img_dset(n, mip)
            """
            files = glob.glob(os.path.join(self.path, f"field_*{n}*.h5"))
            pattern = re.compile(f"field_(\d+)_{name}_MIP(\d){self.field_tag}.h5")

            for f_abs in files:
                f = os.path.basename(f_abs)
                match = pattern.match(f)
                if match is not None:
                    stage = int(match.group(1))
                    mip = int(match.group(2))
                    self.load_field_dset(n, mip, stage)"""

    def load_img_composite_dset(self, mip):
        dset_list = []
        for name in self.names:
            dset_list.append(self.get_img_dset(mip=mip, name=name))
        if len(dset_list) == 0:
            raise Exception(f"No image datasets found at MIP{mip}")

        self.img_composite_dsets[mip] = MyConcatDataset(dset_list)

    def load_field_composite_dset(self, mip, stage):
        dset_list = []
        for name in self.names:
            dset = self.get_field_dset(mip=mip, name=name, stage=stage)
            if dset is None:
                raise Exception(
                    f"No field datasets with tag '{self.field_tag}' found for stage {stage} at MIP{mip} for dataset '{name}'"
                )
            dset_list.append(dset)
        self.field_composite_dsets[mip][stage] = MyConcatDataset(dset_list)

    def get_img_dset(self, name, mip):
        return self.img_dsets[mip][name]

    def load_img_dset(self, name, mip):
        img_file = self.get_img_file(name=name, mip=mip)
        if img_file is None:
            return None

        if "img" in img_file:
            img_dset = img_file["img"]
        else:
            img_dset = img_file["main"]
        self.img_dsets[mip][name] = img_dset

    def get_field_dset(self, name, mip, stage):
        if self.field_dsets[mip][name][stage] is None:
            self.load_field_dset(name=name, mip=mip, stage=stage)
        return self.field_dsets[mip][name][stage]

    def load_field_dset(self, name, mip, stage, shape=None, create=False):
        """Return field datset for name, mip, stage; if not present, create it

        Args:
            name (str)
            mip (int)
            stage (int)
            shape (Tuple(int)): shape of associated image dataset, Nx2xWxH
            create (bool)
        """
        field_file = self.get_field_file(name=name, mip=mip, stage=stage, create=create)
        if field_file is None and not create:
            for prev_mip in range(mip + 1, self.max_mip + 1):
                prev_field_file = self.get_field_file(
                    name=name, mip=prev_mip, stage=stage
                )
                if prev_field_file is not None:
                    self._upsample_field_direct(
                        name=name, stage=stage, in_mip=prev_mip, out_mip=mip
                    )
                    field_file = self.get_field_file(
                        name=name, mip=mip, stage=stage, create=create
                    )
                    break

        if field_file is None:
            return None

        field_dset = None
        if create:
            if "main" in field_file:
                del field_file["main"]
            if "field" in field_file:
                del field_file["field"]

            field_shape = (shape[0], 2, shape[-2], shape[-1])
            chunks = (1, 2, shape[-2], shape[-1])
            field_dset = field_file.create_dataset(
                "main",
                shape=field_shape,
                dtype=np.float32,
                chunks=chunks,
                compression="lzf",
                scaleoffset=2,
            )
        else:
            if "main" in field_file:
                field_dset = field_file["main"]
            elif "field" in field_file:
                field_dset = field_file["field"]

        self.field_dsets[mip][name][stage] = field_dset
        return field_dset

    def get_img_file(self, name, mip):
        path = os.path.join(self.path, f"{name}_MIP{mip}.h5")
        print(f"Loading file '{path}...'")
        if os.path.isfile(path):
            return h5py.File(path, "r")
        else:
            return None

    def get_field_file(self, name, mip, stage, create=False):
        path = os.path.join(
            self.path, f"field_{stage}_{name}_MIP{mip}{self.field_tag}.h5"
        )
        print(f"Loading file '{path}...'")
        if create:
            return h5py.File(path, "a")
        if os.path.isfile(path):
            return h5py.File(path, "r")
        else:
            return None

    def generate_fields(self, model, mip, stage):
        for name in self.names:
            img_dset = self.get_img_dset(name, mip)
            if stage > 0:
                prev_field_file = self.get_field_file(name, mip, stage - 1)
                if prev_field_file is None:
                    raise Exception(
                        "Attempting to generate field for "
                        f"stage {stage} before field for stage {stage-1} "
                        f"has been generated (at MIP{mip})"
                    )
                if "field" in prev_field_file:
                    prev_field_dset = prev_field_file["field"]
                else:
                    prev_field_dset = prev_field_file["main"]

            else:
                prev_field_dset = None

            self.load_field_dset(name, mip, stage, shape=img_dset.shape, create=True)
            field_dset = self.get_field_dset(name, mip, stage)

            self._generate_field_dataset(model, img_dset, field_dset, prev_field_dset)

    def _generate_field_dataset(self, model, img_dset, field_dset, prev_field_dset):
        """Generate predicted source field for src, tgt pairs
        Pairs are permutable, so predict field for src,tgt as well as tgt,src
        """
        for b in range(img_dset.shape[0]):
            src = helpers.to_tensor(img_dset[b, 0])
            tgt = helpers.to_tensor(img_dset[b, 1])
            if prev_field_dset is not None:
                prev_field = helpers.to_tensor(prev_field_dset[b])
            else:
                prev_field = None

            field = model(
                src_img=src,
                tgt_img=tgt,
                src_agg_field=prev_field,
                train=False,
                return_state=False,
            )
            field_dset[b] = helpers.get_np(field)

    def _upsample_field_direct(self, name, stage, in_mip, out_mip):
        """Upsample a field datset from SRC_MIP to DST_MIP

        Args:
            name (str): filename of dataset (input & output)
            stage (int): module which created dataset (input & output)
            in_mip (int): MIP level of the input dataset
            out_mip (int): MIP level of the output dataset
        """
        assert in_mip > out_mip
        in_field_dset = self.get_field_dset(name=name, stage=stage, mip=in_mip)
        dst_img_dset = self.get_img_dset(name=name, mip=out_mip)
        out_field_dset = self.load_field_dset(
            name=name, stage=stage, mip=out_mip, shape=dst_img_dset.shape, create=True
        )
        # scale_factor = 2**(in_mip - out_mip)

        with torch.no_grad():
            hsz = (in_field_dset.shape[-1] - dst_img_dset.shape[-1]) // 2
            for n in range(in_field_dset.shape[0]):
                in_field = helpers.to_tensor(in_field_dset[n : n + 1]).field()
                in_field = in_field * (2 ** in_mip)
                out_field = in_field.up(mips=in_mip - out_mip)
                out_field = out_field / (2 ** out_mip)
                # out_field = torch.nn.functional.interpolate(in_field,
                #                                  mode='bilinear',
                #                                  scale_factor=scale_factor,
                #                                  align_corners=False,
                #                                  recompute_scale_factor=False
                #                                  ) * scale_factor
                out_field_cropped = out_field[0, :, hsz:-hsz, hsz:-hsz]
                out_field_dset[n] = helpers.get_np(out_field_cropped[...])

    def get_alignment_dset(
        self,
        mip,
        stage=None,
        start_index=0,
        end_index=None,
        crop_mode=None,
        cropped_size=None,
    ):
        """Create AlignmentDataLoader providing image pairs with associated fields

        Details about associated fields:
            * If STAGE > 0, then there must be a field available
            * The field available must be at the required MIP of STAGE
            * The field for the current STAGE is indexed at the current MIP,
                but by the previous STAGE (STAGE - 1)
        """

        if self.img_composite_dsets[mip] is None:
            self.load_img_composite_dset(mip)
        if (
            stage not in [None, 0]
            and self.field_composite_dsets[mip][stage - 1] is None
        ):
            self.load_field_composite_dset(mip, stage - 1)

        return AlignmentDataLoader(
            img_dset=self.img_composite_dsets[mip],
            field_dset=self.field_composite_dsets[mip][stage - 1],
            start_index=start_index,
            end_index=end_index,
            crop_mode=crop_mode,
            cropped_size=cropped_size,
            device=self.device,
        )

    def get_train_dset(self, mip, stage=None, crop_mode="random", cropped_size=1024):
        """Get training dataset: all but last two image pairs"""
        self.load_img_composite_dset(mip)
        return self.get_alignment_dset(
            mip=mip,
            stage=stage,
            start_index=0,
            end_index=len(self.img_composite_dsets[mip]) - 2,
            crop_mode=crop_mode,
            cropped_size=cropped_size,
        )

    def get_val_dset(self, mip, stage=None, crop_mode="middle", cropped_size=2048):
        """Get validation dataset: last two images pairs"""
        self.load_img_composite_dset(mip)
        return self.get_alignment_dset(
            mip=mip,
            stage=stage,
            start_index=len(self.img_composite_dsets[mip]) - 2,
            crop_mode=crop_mode,
            cropped_size=cropped_size,
        )


class AlignmentDataLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        img_dset,
        field_dset,
        start_index,
        end_index,
        crop_mode=None,
        cropped_size=None,
        device="cuda",
    ):
        self.img_dset = img_dset
        self.field_dset = field_dset

        self.end_index = len(img_dset)
        self.start_index = 0
        if end_index is not None:
            self.end_index = min(self.end_index, end_index)
        if start_index is not None:
            self.start_index = max(self.start_index, start_index)
        self.shape = self.img_dset[0].shape
        self.crop_mode = crop_mode
        self.cropped_size = cropped_size
        self.device = device

    def __len__(self):
        return self.end_index - self.start_index

    def set_size_limit(self, n):
        if n < len(self):
            self.end_index = self.start_index + n

    def _get_crop_coords(self):
        if self.crop_mode == "random":
            x_bot, x_top, y_bot, y_top = get_random_crop_coords(
                self.shape[-2:], self.cropped_size
            )
        elif self.crop_mode == "middle":
            x_bot, x_top, y_bot, y_top = get_center_crop_coords(
                self.shape[-2:], self.cropped_size
            )
        elif self.crop_mode is None:
            x_bot = y_bot = 0
            x_top = self.shape[-2]
            y_top = self.shape[-1]
        else:
            raise Exception(f"Bad crop mode: '{self.crop_mode}'")

        return x_bot, x_top, y_bot, y_top

    def __getitem__(self, i):
        x_bot, x_top, y_bot, y_top = self._get_crop_coords()
        x_slice = slice(x_bot, x_top)
        y_slice = slice(y_bot, y_top)

        img = self.img_dset[self.start_index + i]
        src = helpers.to_tensor(img[0, x_slice, y_slice], device=self.device)
        tgt = helpers.to_tensor(img[1, x_slice, y_slice], device=self.device)
        src[src < -4] = 0
        tgt[tgt < -4] = 0

        field = None
        if self.field_dset is not None:
            full_field = self.field_dset[self.start_index + i]
            field = full_field[..., x_bot:x_top, y_bot:y_top]
            field = helpers.to_tensor(field, device=self.device)

        bundle = {
            "src": src,
            "tgt": tgt,
            "src_zeros": src == 0,
            "tgt_zeros": tgt == 0,
            "id": helpers.to_tensor(i, device=self.device),
        }
        if field is not None:
            bundle["src_field"] = field
        return bundle


def get_random_crop_coords(full_shape, cropped_size):
    if full_shape[0] < cropped_size:
        x_bot = 0
        x_top = full_shape[0]
    else:
        x_bot = np.random.randint(0, full_shape[0] - cropped_size + 1)
        x_top = x_bot + cropped_size

    if full_shape[1] < cropped_size:
        y_bot = 0
        y_top = full_shape[1]
    else:
        y_bot = np.random.randint(0, full_shape[1] - cropped_size + 1)
        y_top = y_bot + cropped_size

    return x_bot, x_top, y_bot, y_top


def get_center_crop_coords(full_shape, cropped_size):
    if full_shape[0] < cropped_size:
        x_bot = 0
        x_top = full_shape[0]
    else:
        x_bot = (full_shape[0] - cropped_size) // 2
        x_top = x_bot + cropped_size

    if full_shape[1] < cropped_size:
        y_bot = 0
        y_top = full_shape[1]
    else:
        y_bot = (full_shape[1] - cropped_size) // 2
        y_top = y_bot + cropped_size

    return x_bot, x_top, y_bot, y_top
