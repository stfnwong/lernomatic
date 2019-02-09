"""
VIS_LR
Visualization tools for learning rate stuff

Stefan Wong 2019
"""

def plot_lr_vs_acc(ax, lr_data, acc_data, **kwargs):

    title = kwargs.pop('title', 'Learning Rate vs. Accuracy')

    if len(lr_data) != len(acc_data):
        plot_len = min([len(lr_data), len(acc_data)])
    else:
        plot_len = len(lr_data)

    ax.plot(lr_data[:plot_len], acc_data[:plot_len])
    #ax.set_xlim([lr_data[0], lr_data[plot_len]])
    #ax.set_ylim([acc_data[0], acc_data[plot_len]])
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Learning Rate')
    ax.set_title(title)
