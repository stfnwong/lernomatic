"""
VIS_LOSS_HISTORY
Draw a graph of the training / test loss for a loss history item

Stefan Wong 2018
"""

import numpy as np

# debug
#from pudb import set_trace; set_trace()


def smooth_loss(loss_curve, beta=0.98):
    avg_loss = 0.0
    smoothed_loss = np.zeros(len(loss_curve))
    for n, loss in enumerate(loss_curve):
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_loss[n] = avg_loss / (1 - beta ** n+1)

    return smoothed_loss


def plot_train_history(ax, loss_curve, **kwargs):

    test_loss_curve = kwargs.pop('test_loss_curve', None)
    acc_curve       = kwargs.pop('acc_curve', None)
    plot_title      = kwargs.pop('title', 'Loss curve')
    iter_per_epoch  = kwargs.pop('iter_per_epoch', 0)
    cur_epoch       = kwargs.pop('cur_epoch', 0)
    max_ticks       = kwargs.pop('max_ticks', 6)

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

    if test_loss_curve is not None:
        ax.plot(np.linspace(0, 1, len(test_loss_curve)), test_loss_curve)
        legend_entries.append('Test Loss')

    if acc_curve is not None:
        # create a new (scaled) accuracy curve
        acc_ticks = np.linspace(0, 1.0, max_ticks)
        ac_ax = ax.twinx()
        ac_ax.plot(np.linspace(0, 1, len(acc_curve)), acc_curve, 'rx')
        ac_ax.set_xticklabels([])
        ac_ax.set_ylabel('Accuracy')
        ac_ax.set_ylim([0, 1.0])
        ac_ax.set_yticks(acc_ticks)

    # Plot loss
    ax.plot(np.linspace(0, 1, len(loss_curve)), loss_curve)
    legend_entries.append('Loss')

    loss_ticks = np.linspace(np.min(loss_curve), np.max(loss_curve), max_ticks, endpoint=True)
    iter_ticks = np.linspace(0, len(loss_curve), max_ticks+1, endpoint=True)

    ax.set_xlabel('Iteration')
    #ax.set_xticks(iter_ticks)
    ax.set_xticklabels([str(int(i)) for i in iter_ticks])
    ax.set_ylabel('Loss')
    ax.set_yticks(loss_ticks)
    ax.set_ylim([np.min(loss_curve), np.max(loss_curve)])
    ax.set_title(plot_title)
    # Add legend entries
    ax.legend(legend_entries)


def plot_train_history_2subplots(ax, loss_curve, **kwargs):
    if type(ax) is not list:
        raise ValueError('ax must be a list of axes handles')
    if len(ax) < 2:
        raise ValueError('ax list must contain at least 2 axes handles')

    test_loss_curve = kwargs.pop('test_loss_curve', None)
    acc_curve       = kwargs.pop('acc_curve', None)
    plot_title      = kwargs.pop('title', 'Loss curve')
    iter_per_epoch  = kwargs.pop('iter_per_epoch', 0)
    cur_epoch       = kwargs.pop('cur_epoch', 0)
    max_ticks       = kwargs.pop('max_ticks', 6)
    loss_title      = kwargs.pop('loss_title', None)
    acc_title       = kwargs.pop('acc_title', None)

    # plot the loss
    ax[0].plot(np.arange(len(loss_curve)), loss_curve)
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

    if acc_curve is not None:
        ax[1].plot(np.arange(len(acc_curve)), acc_curve, 'r')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Accuracy')
        if acc_title is None:
            ax[1].set_title('Accuracy history')
        else:
            ax[1].set_title(acc_title)
