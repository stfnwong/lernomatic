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

    # Plot loss
    ax.plot(np.arange(len(loss_curve)), loss_curve)
    legend_entries.append('Loss')

    if (cur_epoch > 0) and (iter_per_epoch != 0):
        # Try to add top axis (to show units in epochs rather than
        # iterations)
        epoch_ticks = np.linspace(0, cur_epoch, max_ticks, endpoint=True)
        axx = ax.twiny()
        axx.set_xlim([0, cur_epoch])
        axx.set_xticks(epoch_ticks)
        axx.set_xlabel('Epochs')

    if test_loss_curve is not None:
        test_loss_stretch = np.zeros(len(loss_curve))
        test_loss_range = int(len(loss_curve) / iter_per_epoch)
        for e in range(cur_epoch):
            test_loss_stretch[e: (e+1) * test_loss_range] = test_loss_curve[e]
        ax.plot(np.arange(len(test_loss_stretch)), test_loss_stretch)
        legend_entries.append('Test Loss')
        #ax.plot(np.arange(len(test_loss_curve)), test_loss_curve)

    if acc_curve is not None:
        acc_stretch = np.zeros(len(loss_curve))
        acc_range = int(len(loss_curve) / iter_per_epoch)
        for e in range(cur_epoch):
            acc_stretch[e : (e+1) * acc_range] = acc_curve[e]
        ax.plot(np.arange(len(acc_stretch)), acc_stretch)
        legend_entries.append('Accuracy')
        # Add another axis with accuracy scale
        acc_ticks = np.linspace(0, 1.0, max_ticks)
        ayy = ax.twinx()
        ayy.set_ylabel('Accuracy')
        ayy.set_ylim([0, 1.0])
        ayy.set_yticks(acc_ticks)

    ax.set_xlabel('Iteration')
    loss_ticks = np.linspace(np.min(loss_curve), np.max(loss_curve), max_ticks, endpoint=True)
    ax.set_ylabel('Loss')
    ax.set_yticks(loss_ticks)
    ax.set_ylim([np.min(loss_curve), np.max(loss_curve)])
    ax.set_title(plot_title)
    ax.legend(legend_entries)
