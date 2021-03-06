"""
VIS_LOSS_HISTORY
Draw a graph of the training / test loss for a loss history item

Stefan Wong 2018
"""

import numpy as np
import matplotlib.pyplot as plt

# debug
#

def get_figure_subplots(num_subplots:int=2) -> tuple:
    fig = plt.figure()
    ax = []
    for p in range(num_subplots):
        sub_ax = fig.add_subplot(num_subplots, 1, (p+1))
        ax.append(sub_ax)

    return (fig, ax)


def smooth_loss(loss_history:np.ndarray, beta:float=0.98) -> np.ndarray:
    avg_loss = 0.0
    smoothed_loss = np.zeros(len(loss_history))
    for n, loss in enumerate(loss_history):
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_loss[n] = avg_loss / (1 - beta ** n+1)

    return smoothed_loss


def plot_train_history(ax, loss_history: np.ndarray, **kwargs) -> None:
    test_loss_history : np.ndarray = kwargs.pop('test_loss_history', None)
    acc_history       : np.ndarray = kwargs.pop('acc_history', None)
    plot_title        : str        = kwargs.pop('title', 'Loss curve')
    iter_per_epoch    : int        = kwargs.pop('iter_per_epoch', 0)
    cur_epoch         : int        = kwargs.pop('cur_epoch', 0)
    max_ticks         : int        = kwargs.pop('max_ticks', 6)

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

    if test_loss_history is not None:
        ax.plot(np.linspace(0, 1, len(test_loss_history)), test_loss_history)
        legend_entries.append('Test Loss')

    if acc_history is not None:
        # create a new (scaled) accuracy curve
        acc_ticks = np.linspace(0, 1.0, max_ticks)
        ac_ax = ax.twinx()
        ac_ax.plot(np.linspace(0, 1, len(acc_history)), acc_history, 'rx')
        ac_ax.set_xticklabels([])
        ac_ax.set_ylabel('Accuracy')
        ac_ax.set_ylim([0, 1.0])
        ac_ax.set_yticks(acc_ticks)

    # Plot loss
    ax.plot(np.linspace(0, 1, len(loss_history)), loss_history)
    legend_entries.append('Loss')

    loss_ticks = np.linspace(np.min(loss_history), np.max(loss_history), max_ticks, endpoint=True)
    iter_ticks = np.linspace(0, len(loss_history), max_ticks+1, endpoint=True)

    ax.set_xlabel('Iteration')
    #ax.set_xticks(iter_ticks)
    ax.set_xticklabels([str(int(i)) for i in iter_ticks])
    ax.set_ylabel('Loss')
    ax.set_yticks(loss_ticks)
    ax.set_ylim([np.min(loss_history), np.max(loss_history)])
    ax.set_title(plot_title)
    # Add legend entries
    ax.legend(legend_entries)


def plot_train_history_2subplots(ax, loss_history:np.ndarray, **kwargs) -> None:
    if type(ax) is not list:
        raise ValueError('ax must be a list of axes handles')
    if len(ax) < 2:
        raise ValueError('ax list must contain at least 2 axes handles')

    test_loss_history : np.ndarray = kwargs.pop('test_loss_history', None)
    test_loss_title   : str        = kwargs.pop('test_loss_title', 'Test Loss curve')
    acc_history       : np.ndarray = kwargs.pop('acc_history', None)
    iter_per_epoch    : int        = kwargs.pop('iter_per_epoch', 0)
    cur_epoch         : int        = kwargs.pop('cur_epoch', 0)
    max_ticks         : int        = kwargs.pop('max_ticks', 6)
    loss_title        : str        = kwargs.pop('loss_title', None)
    acc_title         : str        = kwargs.pop('acc_title', None)

    # plot the loss
    ax[0].plot(np.arange(len(loss_history)), loss_history)
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Loss')
    if loss_title is None:
        ax[0].set_title('Loss history')
    else:
        ax[0].set_title(loss_title)

    gen_epoch_ticks = (cur_epoch > 0) and (iter_per_epoch != 0)
    if gen_epoch_ticks:
        # Try to add top axis (to show units in epochs rather than
        # iterations)
        epoch_ticks = np.linspace(0, cur_epoch, max_ticks, endpoint=True)
        epoch_axis = ax[0].twiny()
        epoch_axis.set_xlim([0, cur_epoch])
        epoch_axis.set_xticks(epoch_ticks)
        epoch_axis.set_xlabel('Epochs')

    if acc_history is not None:
        ax[1].plot(np.arange(len(acc_history)), acc_history, 'r')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        if acc_title is None:
            ax[1].set_title('Accuracy history')
        else:
            ax[1].set_title(acc_title)

    if test_loss_history is not None and len(ax) > 2:
        ax[2].plot(np.arange(len(test_loss_history)), test_loss_history)
        ax[2].set_xlabel('Iteration')
        ax[2].set_ylabel('Accuracy')
        ax[2].set_title(test_loss_title)


def plot_train_history_dcgan(ax, g_loss_history, d_loss_history, **kwargs):
    iter_per_epoch    = kwargs.pop('iter_per_epoch', 0)
    cur_epoch         = kwargs.pop('cur_epoch', 0)
    max_ticks         = kwargs.pop('max_ticks', 6)
    loss_title        = kwargs.pop('loss_title', None)
    acc_title         = kwargs.pop('acc_title', None)

    if type(ax) is not list:
        ax = [ax]

    if len(ax) == 1:
        legend_list = ['G loss', 'D loss']
        ax[0].plot(np.arange(len(g_loss_history)), g_loss_history)
        ax[0].plot(np.arange(len(d_loss_history)), d_loss_history)
        ax[0].set_xlabel('Iterations')      # TODO : add epoch axis...
        ax[0].set_ylabel('Loss')

        if loss_title is not None:
            ax[0].set_title(loss_title)
        else:
            ax[0].set_title('DCGAN Loss History')

        gen_epoch_ticks = (cur_epoch > 0) and (iter_per_epoch != 0)
        if gen_epoch_ticks:
            # Try to add top axis (to show units in epochs rather than
            # iterations)
            epoch_ticks = np.linspace(0, cur_epoch, max_ticks, endpoint=True)
            epoch_axis = ax[0].twiny()
            epoch_axis.set_xlim([0, cur_epoch])
            epoch_axis.set_xticks(epoch_ticks)
            epoch_axis.set_xlabel('Epochs')

    # If we have 2 or more axis, plot into the first two and ignore any others
    elif len(ax) >= 2:
        # generator
        ax[0].plot(np.arange(len(g_loss_history)), g_loss_history, 'g')
        ax[0].set_xlabel('Iterations')      # TODO : add epoch axis...
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Generator Loss History')

        # discriminator
        ax[1].plot(np.arange(len(d_loss_history)), d_loss_history, 'b')
        ax[1].set_xlabel('Iterations')      # TODO : add epoch axis...
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Discriminator Loss History')
