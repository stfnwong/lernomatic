"""
VIS_LOSS_HISTORY
Draw a graph of the training / test loss for a loss history item

Stefan Wong 2018
"""

import numpy as np

# debug
#from pudb import set_trace; set_trace()

def plot_loss_history(ax, loss_curve, **kwargs):

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
