"""
LR_EXPERIMENTS
Some experiments on centroid weighting

Stefan Wong 2019
"""

import numpy as np
import matplotlib.pyplot as plt

from lernomatic.param import lr_common


def single_centroid(X:np.ndarray) -> np.float64:
    # NOTE: this method sucks, just here for completeness
    # Return the 1d centroid of the data
    xm = np.sum((np.arange(len(X)) * X) / np.sum(X))

    return xm


def sum_of_centroid_segments(X:np.ndarray) ->  np.ndarray:
    pass


def main() -> None:
    # get a finder from disk
    finder_state_file = '[resnet][max_acc]_lr_finder_state.pth'
    lr_finder = lr_common.lr_finder_auto_load(finder_state_file)

    print('Loaded lr_finder from file [%s]' % str(finder_state_file))
    print(lr_finder)

    # lets clone the accurracy history
    acc_history = np.asarray(lr_finder.acc_history)


    # make a plot
    fig, ax = plt.subplots()

    ax.plot(acc_history)
    # plot the single centroid
    xm = single_centroid(acc_history)
    ax.axvline(x = xm, color='r')

    ax.set_title('Accuracy vs. iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    fig.tight_layout()
    fig.set_size_inches(10, 10)

    plt.show()


if __name__ == '__main__':
    main()
