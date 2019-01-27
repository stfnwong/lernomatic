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
        #test_loss_stretch = np.zeros(len(loss_curve))
        #test_loss_ratio = int(len(loss_curve) / len(test_loss_curve))
        #for e in range(len(test_loss_curve)):
        #    test_loss_stretch[e: (e+1) * test_loss_ratio] = test_loss_curve[e]
        #ax.plot(np.arange(len(test_loss_stretch)), test_loss_stretch)
        tl_ax = ax.twiny()
        tl_ax.plot(np.arange(len(test_loss_curve)), test_loss_curve, 'g')
        tl_ax.set_xticklabels([])
        #tl_ax.legend('Test Loss')

    if acc_curve is not None:
        #acc_stretch = np.zeros(iter_per_epoch * cur_epoch)
        #for e in range(len(acc_curve)):
        #    acc_stretch[e : (e+1) * iter_per_epoch] = acc_curve[e]
        #ax.plot(np.arange(len(acc_stretch)), acc_stretch)
        #legend_entries.append('Accuracy')
        acc_ticks = np.linspace(0, 1.0, max_ticks)
        ax.plot(np.arange(len(acc_curve)), acc_curve, 'rx')
        ac_ax = ax.twiny()
        ac_ax.plot(np.arange(len(acc_curve)), acc_curve, 'rx')
        ac_ax.set_xticklabels([])
        ac_ax.set_ylabel('Accuracy')
        ac_ax.set_ylim([0, 1.0])
        ac_ax.set_yticks(acc_ticks)

        #ac_ax.set_yticklabels([])
        # Add another axis with accuracy scale
        #ayy = ax.twinx()
        #ayy.set_ylabel('Accuracy')
        #ayy.set_ylim([0, 1.0])
        #ayy.set_yticks(acc_ticks)
        #ac_ax.legend('Accuracy')

    ax.plot(np.arange(len(loss_curve)), loss_curve)
    #legend_entries.append('Loss')
    ax.set_xlabel('Iteration')
    loss_ticks = np.linspace(np.min(loss_curve), np.max(loss_curve), max_ticks, endpoint=True)
    ax.set_ylabel('Loss')
    ax.set_yticks(loss_ticks)
    ax.set_ylim([np.min(loss_curve), np.max(loss_curve)])
    ax.set_title(plot_title)
    ax.legend('Loss')
