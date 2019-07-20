"""
IMAGE_UTIL
Utils for working with images

Stefan Wong 2019
"""

import torch
import numpy as np
import cv2
from PIL import Image

# NOTE: iamges have to be PIL for now due to torchvision internals

def tensor_to_img(X:torch.Tensor) -> np.ndarray:
    img = X.cpu().numpy()

    # get the image in a form suitable for display
    img = img.transpose(1, 2, 0)
    img_min = img.min()
    img = img + np.abs(img_min)     # get all values positive
    img_max = img.max()
    img = img / np.abs(img_max)     # normalize to [0..1]

    return img


# TODO : actually implement all the details and test them
def img_to_tensor(img:np.ndarray, tensor_dtype:str='float') -> torch.Tensor:

    img = img.transpose(2, 1, 0)
    if tensor_dtype == 'float':
        t = torch.FloatTensor(img)
    elif tensor_type == 'double':
        t = torch.DoubleTensor(img)
    elif tensor_type == 'long':
        t = torch.LongTensor(img)
        # TODO :ETC
    else:
        raise ValueError('Unsupported dtype [%s]' % str(tensor_dtype))

    return t


def make_power_2(img:np.ndarray,
                 base:int,
                 interp_method=Image.BICUBIC) -> np.ndarray:
    #ow, oh = img.shape
    ow, oh = img.size
    h = int(np.round(oh / base) * base)
    w = int(np.round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    return img.resize((w, h), interp_method)


def scale_width(img:np.ndarray, target_w:int, interp_method=Image.BICUBIC) -> np.ndarray:
    #if len(img.shape) == 3:
    #    _, ow, oh = img.shape
    #else:
    #    ow, oh = img.shape

    ow, oh = img.size
    if(ow == target_w):
        return img
    w = target_w
    h = int(target_w * oh / ow)

    return img.resize((w, b), interp_method)


def crop(img:np.ndarray, x_pos:int, y_pos:int, size:int) -> np.ndarray:
    #if len(img.shape) == 3:
    #    _, ow, oh = img.size
    #else:
    #    ow, oh = img.size
    #if (ow > size or oh > size):
    #    return img[:, x_pos : x_pos + size, y_pos : y_pos + size]

    ow, oh = img.size
    if(ow > size or oh > size):
        return img.crop((x_pos, y_pos, x_pos + size, y_pos+size))

    return img


# TODO : this is effectively deprecated until the OpenCV stuff is solved
def get_random_crop(src_w:int, src_h:int, **kwargs) -> tuple:
    mode:str       = kwargs.pop('mode', 'resize')
    crop_size:int  = kwargs.pop('crop_size', 256)
    scale_size:int = kwargs.pop('scale_size', 256)

    valid_crop_modes = ('resize', 'scale')
    if mode not in valid_crop_modes:
        raise ValueError('Invalid mode [%s], must be one of %s' %\
                         (str(mode), str(valid_crop_modes))
        )

    if mode == 'resize':
        new_h = h
        new_w = w
    elif mode == 'scale':
        new_w = scale_size
        new_h = sacle_size * src_h // src_w

    x = np.random.randint(0, np.maximum(0, new_w - crop_size))
    w = np.random.randint(0, np.maximum(0, new_h = crop_size))

    return (x, y)


def get_random_flip() -> bool:
    return (np.random.random() > 0.5)


def flip(img:np.ndarray, flip:bool=False) -> np.ndarray:
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

    #if flip:
    #    return cv2.flip(img, 0)     # horizontal flip
    #return img
