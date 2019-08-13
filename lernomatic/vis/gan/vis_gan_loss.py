"""
VIS_GAN_LOSS
Loss history visualizer for a GAN model

Stefan Wong 2019
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_gan_loss(ax, gen_loss:np.ndarray, disc_loss:np.ndarray, **kwargs) -> None:

    iter_per_epoch    : int        = kwargs.pop('iter_per_epoch', 0)
    cur_epoch         : int        = kwargs.pop('cur_epoch', 0)
    max_ticks         : int        = kwargs.pop('max_ticks', 6)
    loss_title        : str        = kwargs.pop('loss_title', None)

    # plot the loss
    legend_entries = []
    ax.plot(np.arange(len(gen_loss)), gen_loss)
    legend_entries.append('Generator Loss')
    ax.plot(np.arange(len(disc_loss)), disc_loss)
    legend_entries.append('Discriminator Loss')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')

    gen_epoch_ticks = (cur_epoch > 0) and (iter_per_epoch != 0)
    if gen_epoch_ticks:
        # Try to add top axis (to show units in epochs rather than
        # iterations)
        epoch_ticks = np.linspace(0, cur_epoch, max_ticks, endpoint=True)
        epoch_axis = ax.twiny()
        epoch_axis.set_xlim([0, cur_epoch])
        epoch_axis.set_xticks(epoch_ticks)
        epoch_axis.set_xlabel('Epochs')

    if loss_title is None:
        ax.set_title('Loss history')
    else:
        ax.set_title(loss_title)

    ax.legend(legend_entries)
