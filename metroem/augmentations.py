import torch
import torchvision

import numpy as np
import skimage

import six
import copy


class Augmentor():
    def __init__(self, augmentations=[]):
        transform = []
        for a_spec in augmentations:
            t = a_spec['type']
            del a_spec['type']

            if t == "warp":
                transform.append(RandomWarp(**a_spec))
            elif t =="random_transpose":
                transform.append(RandomTranspose(**a_spec))
            elif t =="random_src_tgt_swap":
                transform.append(RandomSrcTgtSwap(**a_spec))


        self.transform = torchvision.transforms.Compose(transform)

    def __call__(self, bundle):
        return self.transform(bundle)

class RandomWarp(object):
    """ Warp With Random Field
    difficulty
        0 == one direction
        1 == 2 directions along each axis
        i == 2^i directions along each axis
    """
    def __init__(self, difficulty=2, max_disp=10, prob=1.0, min_disp=0,
                 randomize_d=False, random_epicenters=True):
        assert difficulty >= 0
        self.randomize_d = randomize_d
        self.difficulty = difficulty
        self.max_disp = max_disp
        self.min_disp = min_disp
        self.prob     = prob
        self.random_epicenters = random_epicenters

    def __call__(self, bundle):
        src = bundle['src']

        if self.randomize_d and self.difficulty > 0:
            curr_diff = random.randint(0, self.difficulty)
        else:
            curr_diff = self.difficulty
        coin = np.random.uniform()
        if coin < self.prob:
            # NOTE image dimention assumed to be power of 2

            granularity = int(np.log2(src.shape[-1]) - curr_diff)
            res_delta = generate_random_residuals(src.squeeze().shape,
                                                  min_disp=self.min_disp,
                                                  max_disp=self.max_disp,
                                                  granularity=granularity,
                                                  random_epicenters=self.random_epicenters,
                                                  device=src.device)
            for k, v in six.iteritems(bundle):
                if 'field' in k:
                    #print ("Warp does nothing to {}".format(k))
                    pass
                elif 'tgt' in k:
                    #print (f"Warping {k} with {res_delta[0, 0].mean(), res_delta[0, 1].mean()}")
                    original_dtype = v.dtype
                    v = v.float()
                    v_aug = res_delta.from_pixels()(v)
                    if original_dtype == torch.bool:
                        v_aug = v_aug > 0.4
                    else:
                        zero_mask = v == 0
                        zero_mask_aug = res_delta.from_pixels()(zero_mask.float()) > 0.4
                        v_aug[zero_mask_aug] = 0
                    bundle[k] = v_aug
        return bundle

def generate_random_residuals(shape, max_disp, min_disp=0, granularity=9,
                              random_epicenters=True, device='cuda'):
    if random_epicenters:
        seed_shape = [i // (2**(granularity - 1)) for i in shape]
        up_shape = [i * 2 for i in shape]
    else:
        seed_shape = [i // (2**(granularity)) for i in shape]
        up_shape = shape
    seed_x = np.random.uniform(size=seed_shape, low=min_disp, high=max_disp)
    seed_y = np.random.uniform(size=seed_shape, low=min_disp, high=max_disp)

    up_x = skimage.transform.resize(seed_x, up_shape)
    up_y = skimage.transform.resize(seed_y, up_shape)

    final_x, final_y = random_crop([up_x, up_y], shape)
    result = torch.tensor(np.stack([final_x, final_y], axis=0), device=device)
    result = result.unsqueeze(0).float().field()
    return result

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

class RandomTranspose(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, bundle):
        coin = np.random.uniform()
        if coin < self.prob:
            for k, v in six.iteritems(bundle):
                if 'field' in k:
                    bundle[k] = bundle[k].transpose(2,3).flip(2)
                elif 'id' in k:
                    pass
                else:
                    bundle[k] = v.transpose(-1, -2)

        return bundle

class RandomSrcTgtSwap(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, bundle):
        result = copy.deepcopy(bundle)
        coin = np.random.uniform()
        if coin < self.prob:
            for k, v in six.iteritems(bundle):
                if 'src' in k:
                    result[k.replace('src', 'tgt')] = bundle[k]
                if 'tgt' in k:
                    result[k.replace('tgt', 'src')] = bundle[k]

        return result





