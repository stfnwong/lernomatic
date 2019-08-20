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


def dcgan_gen_image_grid(gen_model:common.LernomaticModel, **kwargs) -> list:

    fixed_noise = kwargs.pop('fixed_noise', None)
    nz          = kwargs.pop('nz', 100)
    img_size    = kwargs.pop('img_size', 64)
    num_images  = kwargs.pop('num_images', 64)

    if fixed_noise is None:
        fixed_noise = torch.randn(img_size, nz, 1, 1)

    img_list = []
    gen_model.eval()

    for n in range(num_images):
        fake_g = gen_model(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake_g, padding=2, normalize=True))

    return img_list



# Utils for sampling from latent space
def midpoint_linear_interp(points:np.ndarray, ix:float) -> Union[float, np.ndarray]:
    n_points = len(points)
    next_ix = (ix + 1) % n_points
    avg = (points[ix] + points[next_ix]) / 2.0

    return np.linalg.norm(avg)


# Mostly taken from https://github.com/soumith/dcgan.torch/issues/14
def slerp(points:np.ndarray, q1:float, q2:float) -> np.ndarray:
    omega = np.arccos(
        np.clip(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)), -1, 1)
    )
    so = np.sin(omega)
    if so == 0.0:
        return (1.0 - points) * q1 + points + q2    # L'Hopital's rule
    return np.sin((1.0 - points) * omega) / so * q1 + np.sin(points * omega) / so * q2
