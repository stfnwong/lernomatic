"""
VIS_LOSS_HISTORY
Draw a graph of the training / test loss for a loss history item

Stefan Wong 2018
"""

import numpy as np

# debug
#from pudb import set_trace; set_trace()

def plot_loss_history(ax, loss_curve, **kwargs):

    acc_curve      = kwargs.pop('acc_curve', None)
    plot_title     = kwargs.pop('title', 'Loss curve')
    iter_per_epoch = kwargs.pop('iter_per_epoch', 0)
    cur_epoch      = kwargs.pop('cur_epoch', 0)
    max_ticks      = kwargs.pop('max_ticks', 6)

    if acc_curve is not None:
        if (cur_epoch > 0) and (iter_per_epoch != 0):

            acc_stretch = np.zeros(len(loss_curve))
            acc_range = int(len(loss_curve) / iter_per_epoch)
            for e in range(cur_epoch):
                acc_stretch[e : (e+1) * acc_range] = acc_curve[e]

            ax.plot(np.arange(len(loss_curve)), loss_curve)
            ax.plot(np.arange(len(acc_stretch)), acc_stretch)
            #ax.plot(np.arange(cur_epoch), acc_curve[0 : cur_epoch])
            ax.legend(['Training loss', 'Validation Accuracy'])
            # Try to add top axis (to show units in epochs rather than
            # iterations)
            epoch_ticks = np.linspace(0, cur_epoch, max_ticks, endpoint=True)
            axx = ax.twiny()
            axx.set_xlim([0, cur_epoch])
            axx.set_xticks(epoch_ticks)
            axx.set_xlabel('Epochs')
        else:
            ax.plot(np.arange(len(loss_curve)), loss_curve, \
                    np.arange(len(acc_curve)), acc_curve)
            ax.legend(['Training loss', 'Validation Accuracy'])
        # twin x axis common to any plot with acc data
        acc_ticks = np.linspace(0, 1.0, max_ticks)
        ayy = ax.twinx()
        ayy.set_ylabel('Accuracy')
        ayy.set_ylim([0, 1.0])
        ayy.set_yticks(acc_ticks)
    else:
        ax.plot(np.arange(len(loss_curve)), loss_curve)

    ax.set_xlabel('Iteration')
    loss_ticks = np.linspace(np.min(loss_curve), np.max(loss_curve), max_ticks, endpoint=True)
    ax.set_ylabel('Loss')
    ax.set_yticks(loss_ticks)
    ax.set_ylim([np.min(loss_curve), np.max(loss_curve)])
    ax.set_title(plot_title)
