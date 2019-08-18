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


# Utils for sampling from latent space
def midpoint_linear_interp(points:np.ndarray, ix:float) -> Union[float, np.ndarray]:
    n_points = len(points)
    next_ix = (ix + 1) % n_points
    avg = (points[ix] + points[next_ix]) / 2.0

    return np.linalg.norm(avg)


def lerp(r:float, p1:float, p2:float) -> float:
    return (1.0 - r) * p1 + r * p2


# Mostly taken from https://github.com/soumith/dcgan.torch/issues/14
def slerp(val:float, q1:np.ndarray, q2:np.ndarray) -> np.ndarray:
    omega = np.arccos(
        np.clip(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)), -1, 1)
    )
    so = np.sin(omega)

    if so == 0.0:
        return lerp(val, q1, q2)
    return np.sin((1.0 - val) * omega) / so * q1 + np.sin(val * omega) / so * q2


def interp_walk(p1:torch.Tensor,
                  p2:torch.Tensor,
                  num_points:int=16,
                  mode:str='linear') -> np.ndarray:
    ratios = np.linspace(0, 1, num=num_points)
    vectors = list()

    if mode == 'linear':
        for r in ratios:
            v = lerp(r, p1, p2)
            vectors.append(v)
    elif mode == 'spherical' or mode == 'sphere':
        for r in ratios:
            v = slerp(r, p1, p2)
            vectors.append(v)
    else:
        raise ValueError('Invalid interpolation [%s]' % str(mode))

    return np.asarray(vectors)

# TODO : more complex walks around latent space
