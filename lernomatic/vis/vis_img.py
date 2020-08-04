"""
VIS_IMG
Helper functions for visualizing images

Stefan Wong 2019
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from lernomatic.util import image_util


def get_grid_subplots(num_x:int, num_y:int=0) -> tuple:
    if num_y == 0:
        num_y = num_x

    fig = plt.figure()
    ax = []

    num_subplots = num_x * num_y
    for p in range(1, num_subplots+1):
        sub_ax = fig.add_subplot(num_x, num_y, p)
        ax.append(sub_ax)

    return (fig, ax)


def plot_tensor_batch(ax, image_batch:torch.Tensor) -> None:

    if len(ax) != image_batch.shape[0]:
        raise ValueError('Number of image axes (%d) does not equal batch size (%d)' %\
                         (len(ax), image_batch.shape[0])
        )

    for img_idx in range(image_batch.shape[0]):
        img = image_util.tensor_to_img(image_batch[img_idx, :, :, :])
        if img.shape[-1] == 1:
            img = img.squeeze(len(img.shape)-1)
        ax[img_idx].imshow(img)
        ax[img_idx].get_xaxis().set_visible(False)
        ax[img_idx].get_yaxis().set_visible(False)


def plot_tensor(ax, image:torch.Tensor) -> None:
    img = image_util.tensor_to_img(image)
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
