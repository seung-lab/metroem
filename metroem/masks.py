import torch
import torchfields

import scipy


def get_prewarp_mask(bundle, keys_to_apply, inv=False):
    if inv == True:
        raise ValueError("inv is not supported!")

    prewarp_result = torch.ones(
        (1, bundle["src"].shape[-2], bundle["src"].shape[-1]),
        device=bundle["src"].device,
    )

    for settings in keys_to_apply:

        name = settings["name"]

        name_in_bundle = bundle.get(name)
        if name_in_bundle != None:
            mask = name_in_bundle.squeeze()

            fm_in_settings = settings.get("fm")
            bin_in_settings = settings.get("binarization")
            mask_value_in_settings = settings.get("mask_value")
            coarsen_ranges_in_settings = settings.get("coarsen_ranges")

            if fm_in_settings != None and len(mask.shape) > 2:
                mask = mask[fm_in_settings : fm_in_settings + 1]

            while len(mask.shape) < len(prewarp_result.shape):
                mask = mask.unsqueeze(0)

            if bin_in_settings != None:
                mask = binarize(mask, bin_in_settings)

            if mask_value_in_settings != None:
                prewarp_result = torch.where(
                    mask != 1.0,
                    torch.tensor(mask_value_in_settings, device=prewarp_result.device),
                    prewarp_result,
                )
            else:
                prewarp_result = torch.where(
                    mask != 1.0,
                    torch.zeros(1, device=prewarp_result.device),
                    prewarp_result,
                )

            around_the_mask = mask
            if coarsen_ranges_in_settings != None:

                for length, weight in coarsen_ranges_in_settings:
                    if length > 10:
                        around_the_mask = coarsen_mask(around_the_mask, length)
                    else:
                        around_the_mask = coarsen_mask(around_the_mask, length)
                    prewarp_result = torch.where(
                        (around_the_mask != 1.0) * (prewarp_result == 1.0),
                        torch.tensor(
                            weight,
                            dtype=prewarp_result.dtype,
                            device=prewarp_result.device,
                        ),
                        prewarp_result,
                    )

    return prewarp_result


def warp_prewarp_mask(prewarp_result, res, do_nothing=False, inv=False):
    if do_nothing:
        result = prewarp_result
    else:
        if ((res != 0).sum() > 0) and not inv:
            result = res.from_pixels()(prewarp_result)
        else:
            result = prewarp_result
    return result


def binarize(img, bin_setting):
    if bin_setting["strat"] == "eq":
        result = img == bin_setting["value"]
    elif bin_setting["strat"] == "neq":
        result = img != bin_setting["value"]
    elif bin_setting["strat"] == "lt":
        result = img < bin_setting["value"]
    elif bin_setting["strat"] == "gt":
        result = img > bin_setting["value"]
    elif bin_setting["strat"] == "between":
        result = (img > bin_setting["range"][0]) * (img < bin_setting["range"][0])
    elif bin_setting["strat"] == "not_between":
        result = ((img < bin_setting["range"][0]) + (img > bin_setting["range"][0])) > 0
    return result.float()


def get_mse_and_smoothness_masks(bundle, mse_keys_to_apply, sm_keys_to_apply, **kwargs):
    if "src" not in mse_keys_to_apply:
        mse_keys_to_apply["src"] = []
    if "tgt" not in mse_keys_to_apply:
        mse_keys_to_apply["tgt"] = []
    if "src" not in sm_keys_to_apply:
        sm_keys_to_apply["src"] = []
    if "tgt" not in sm_keys_to_apply:
        sm_keys_to_apply["tgt"] = []
    return get_warped_srctgt_mask(bundle, mse_keys_to_apply, sm_keys_to_apply)


def get_warped_srctgt_mask(bundle, mse_keys_to_apply, sm_keys_to_apply):
    src_shape = bundle["src"].shape
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

    mask_mse = torch.ones(mask_shape, device=bundle["src"].device)
    mask_sm = torch.ones(mask_shape, device=bundle["src"].device)
    pred_res = bundle["pred_res"]

    if bundle.get("pred_res_zeros") == None:
        bundle["pred_res_zeros"] = torch.zeros_like(pred_res)

    if "src" in mse_keys_to_apply:
        if bundle.get("src_mask_mse_prewarp") == None:
            bundle["src_mask_mse_prewarp"] = get_prewarp_mask(
                bundle, mse_keys_to_apply["src"]
            )
        src_mask_mse = warp_prewarp_mask(bundle["src_mask_mse_prewarp"], pred_res)
        mask_mse *= src_mask_mse

    if "tgt" in mse_keys_to_apply:
        if bundle.get("tgt_mask_mse_prewarp") == None:
            bundle["tgt_mask_mse_prewarp"] = get_prewarp_mask(
                bundle, mse_keys_to_apply["tgt"]
            )
        tgt_mask_mse = warp_prewarp_mask(
            bundle["tgt_mask_mse_prewarp"], bundle["pred_res_zeros"], do_nothing=True
        )
        mask_mse *= tgt_mask_mse

    if "src" in sm_keys_to_apply:
        if bundle.get("src_mask_sm_prewarp") == None:
            bundle["src_mask_sm_prewarp"] = get_prewarp_mask(
                bundle, sm_keys_to_apply["src"]
            )
        src_mask_sm = warp_prewarp_mask(bundle["src_mask_sm_prewarp"], pred_res)
        mask_sm *= src_mask_sm

    if "tgt" in sm_keys_to_apply:
        if bundle.get("tgt_mask_sm_prewarp") == None:
            bundle["tgt_mask_sm_prewarp"] = get_prewarp_mask(
                bundle, sm_keys_to_apply["tgt"]
            )
        tgt_mask_sm = warp_prewarp_mask(
            bundle["tgt_mask_sm_prewarp"], bundle["pred_res_zeros"], do_nothing=True
        )
        mask_sm *= tgt_mask_sm

    if "src_tgt_comb" in sm_keys_to_apply:
        if bundle.get("src_comb_mask_sm_prewarp") == None:
            bundle["src_comb_mask_sm_prewarp"] = get_prewarp_mask(
                bundle, sm_keys_to_apply["src_tgt_comb"]["src"]
            )
        if bundle.get("tgt_comb_mask_sm_prewarp") == None:
            bundle["tgt_comb_mask_sm_prewarp"] = get_prewarp_mask(
                bundle, sm_keys_to_apply["src_tgt_comb"]["tgt"]
            )
        src_comb_mask_sm = warp_prewarp_mask(
            bundle["src_comb_mask_sm_prewarp"],
            bundle["pred_res_zeros"],
            do_nothing=True,
        )
        tgt_comb_mask_sm = warp_prewarp_mask(
            bundle["tgt_comb_mask_sm_prewarp"],
            bundle["pred_res_zeros"],
            do_nothing=True,
        )
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
    result = (img >= low_cutoff) * (img <= high_cutoff)
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
    kernel = torch.ones([1, 1, 3, 3], device=mask.device)
    for _ in range(n):
        if flip:
            mask = mask.logical_not().float()
        mask = (
            torch.nn.functional.conv2d(mask.unsqueeze(1), kernel, padding=1) > 1
        ).squeeze(1)
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
