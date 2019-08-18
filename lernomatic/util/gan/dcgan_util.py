"""
DCGAN_UTIL
Various functions for use with DCGAN

Stefan Wong 2019
"""

from typing import Union
import numpy as np
import torch
import torchvision.utils as vutils

from lernomatic.models import common


# TODO : deprecate this....?
def dcgan_gen_image_grid(gen_model:common.LernomaticModel, **kwargs) -> list:

    fixed_noise = kwargs.pop('fixed_noise', None)
    nz          = kwargs.pop('nz', 128)
    img_size    = kwargs.pop('img_size', 64)
    num_images  = kwargs.pop('num_images', 64)

    if fixed_noise is None:
        fixed_noise = torch.randn(img_size, nz, 1, 1)

    img_list = []
    gen_model.eval()

    for n in range(num_images):
        fake_g = gen_model.forward(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake_g, padding=2, normalize=True))

    return img_list



# Utils for sampling from latent space
def midpoint_linear_interp(points:np.ndarray, ix:float) -> Union[float, np.ndarray]:
    n_points = len(points)
    next_ix = (ix + 1) % n_points
    avg = (points[ix] + points[next_ix]) / 2.0

    return np.linalg.norm(avg)


# Mostly taken from https://github.com/soumith/dcgan.torch/issues/14
def slerp(val:float, q1:np.ndarray, q2:np.ndarray) -> np.ndarray:
    omega = np.arccos(
        np.clip(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)), -1, 1)
    )
    so = np.sin(omega)

    if so == 0.0:
        return (1.0 - val) * q1 + val + q2
    return np.sin((1.0 - val) * omega) / so * q1 + np.sin(val * omega) / so * q2


def interp_walk(p1:np.ndarray, p2:np.ndarray, num_steps:int=10) ->  np.ndarray:
    pass

