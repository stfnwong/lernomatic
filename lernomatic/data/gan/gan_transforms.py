"""
GAN_TRANSFORMS
Get data transforms for use with CycleGAN

Stefan Wong 2019
"""

import cv2
import torch
import torchvision.transforms as transforms
from lernomatic.util import image_util


def get_gan_transforms(**kwargs):
    # collect up keyword args
    do_crop:bool        = kwargs.pop('do_crop', False)
    do_resize:bool      = kwargs.pop('do_resize', False)
    do_scale_width:bool = kwargs.pop('do_scale_width', False)
    do_flip:bool        = kwargs.pop('do_flip', False)
    to_tensor:bool      = kwargs.pop('to_tensor', False)
    grayscale:bool      = kwargs.pop('grayscale', False)
    crop_pos:tuple      = kwargs.pop('crop_pos', None)
    crop_size:int       = kwargs.pop('crop_size', 256)
    scale_size:int      = kwargs.pop('scale_size', 256)
    method              = kwargs.pop('method', cv2.INTER_CUBIC)

    transform_list = []

    if grayscale:
        transform_list += [transforms.Grayscale(1)]

    if do_crop:
        if crop_pos is None:
            transform_list += [transforms.RandomCrop(crop_size)]
        else:
            transform_list += [
                transforms.Lambda(
                    lambda img: image_util.crop(img, crop_pos, crop_size)
                )
            ]

    if do_resize:
        out_size = [scale_size, scale_size]
        transform_list += [transforms.Resize(out_size, method)]

    if do_scale_width:
        transform_list += [transforms.Lambda(
            lambda img: image_util.scale_width(
                img, scale_size, method)
            )
        ]

    if do_flip:
        transform_list += [transforms.Lambda(
            lambda img: image_util.make_power_2(
                img, base=4, method=method)
            )
        ]
    else:
        transform_list += [transforms.RandomHorizontalFlip()]

    if to_tensor:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

    return transforms.Compose(transform_list)
