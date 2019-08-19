"""
VIS_AAE_LOSS_HISTORY
Visualize loss history for an Adversarial Autoencoder

Stefan Wong 2019
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_aae_train_history(ax, g_loss_history:np.ndarray, d_loss_history:np.ndarray, **kwargs) -> None:
    class_loss_history     :np.ndarray = kwargs.pop('class_loss_history', None)
    train_val_loss_history :np.ndarray = kwargs.pop('train_val_loss_history', None)
    plot_title        : str            = kwargs.pop('title', 'Loss curve')
    iter_per_epoch    : int            = kwargs.pop('iter_per_epoch', 0)
    cur_epoch         : int            = kwargs.pop('cur_epoch', 0)
    max_ticks         : int            = kwargs.pop('max_ticks', 6)

    if len(g_loss_history) != len(d_loss_history):
        raise ValueError('length d_loss_history (%d items) != length g_loss_history (%d items)' %\
                         (len(g_loss_history), len(d_loss_history))
        )

    legend_entries = []
    gen_epoch_ticks = (cur_epoch > 0) and (iter_per_epoch != 0)
    if gen_epoch_ticks:
        # Try to add top axis (to show units in epochs rather than
        # iterations)
        epoch_ticks = np.linspace(0, cur_epoch, max_ticks, endpoint=True)
        epoch_axis = ax.twiny()
        epoch_axis.set_xlim([0, cur_epoch])
        epoch_axis.set_xticks(epoch_ticks)
        epoch_axis.set_xlabel('Epochs')

    # Plot the generator and discriminator lossese
    ax.plot(np.linspace(0, 1, len(g_loss_history)), g_loss_history)
    legend_entries.append('Generator loss')
    ax.plot(np.linspace(0, 1, len(d_loss_history)), d_loss_history)
    legend_entries.append('Discriminator loss')

    loss_ticks = np.linspace(np.min(g_loss_history), np.max(g_loss_history), max_ticks, endpoint=True)
    iter_ticks = np.linspace(0, len(g_loss_history), max_ticks+1, endpoint=True)

    ax.set_xlabel('Iteration')
    #ax.set_xticks(iter_ticks)
    ax.set_xticklabels([str(int(i)) for i in iter_ticks])
    ax.set_ylabel('Loss')
    ax.set_yticks(loss_ticks)
    ax.set_ylim([np.min(d_loss_history), np.max(g_loss_history)])
    ax.set_title(plot_title)
    # Add legend entries
    ax.legend(legend_entries)
