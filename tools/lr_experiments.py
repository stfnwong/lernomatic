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
    #xm = np.sum((np.arange(len(X)) * X) / np.sum(X))
    xm = np.sum(np.dot(np.arange(len(X)), X) / np.sum(X))

    return xm


def centroid_segments(X:np.ndarray, num_segments:int=8) ->  np.ndarray:
    seg_len = int(len(X) // num_segments)
    print('Working with %d segments each of length %d' % (num_segments, seg_len))

    centroids = np.zeros(num_segments)
    for seg in range(num_segments):
        seg_start = seg * seg_len
        seg_end = (seg+1) * seg_len
        xm = single_centroid(X[seg_start : seg_end])

        centroids[seg] = xm + (seg * seg_len)

    return (centroids, seg_len)     # TODO : get this out from here so I can compare the centroids to the 'regular' divisions


def sum_segments(X:np.ndarray, num_segments:int=8) -> np.ndarray:
    seg_len = int(len(X) // num_segments)
    print('Working with %d segments each of length %d' % (num_segments, seg_len))

    segment_sums = np.zeros(num_segments)
    for seg in range(num_segments):
        seg_start = seg * seg_len
        seg_end = (seg+1) * seg_len
        segment_sums[seg] = np.sum(X[seg_start : seg_end])

    return (segment_sums, seg_len)     # TODO : get this out from here so I can compare the centroids to the 'regular' divisions


# ======== KDE ======== #

# Normal density
def phi(X:np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.square(X)) / np.sqrt(2 * np.pi)

# h is bandwidth parameter
def kernel(domain:np.ndarray, data:np.ndarray, h:float) -> np.ndarray:
    return phi((domain - data) / h) / h
    #return np.mean(phi((domain - data) / h) / h)

#def kernel_elem(X:np.ndarray, data:np.ndarray, h:float) -> np.ndarray:
#    pass

def kernel_density_est(data:np.ndarray) -> np.ndarray:

    # silverman bandwidth
    h = np.std(data) * np.power(((4 / 3) * np.prod(data.shape)), 1/5)
    #domain = np.arange(len(data))
    domain = np.linspace(0, len(data), len(data))

    return kernel(domain, data, h)



# TODO : the idea here is encapsulate each 'method' in its own function

def plot_centroid_segments(ax, X:np.ndarray, centroids:np.ndarray, seg_len:int) -> None:
    # First plot the 'natural' segment boundaries
    for s in range(len(centroids)):
        ax.axvline(x = (s * seg_len), color='g')

    for n, xm in enumerate(centroids):
        print('Centroid %d : %f' % (n, xm))
        ax.axvline(x = xm, color='r')

    ax.plot(X)



def main() -> None:
    # get a finder from disk
    finder_state_file = '[resnet][max_acc]_lr_finder_state.pth'
    lr_finder = lr_common.lr_finder_auto_load(finder_state_file)

    print('Loaded lr_finder from file [%s]' % str(finder_state_file))
    print(lr_finder)

    # lets clone the accurracy history
    #acc_history = np.asarray(lr_finder.acc_history[400 : 800])
    acc_history = np.asarray(lr_finder.acc_history)

    # what is the sum of the bottom half vs the top half?
    lower_sum = np.sum(acc_history[ : len(acc_history) // 2])
    upper_sum = np.sum(acc_history[len(acc_history) // 2 : ])

    print('Sum of acc_history[0 : %d]  : %f' % (len(acc_history) // 2, lower_sum))
    print('Sum of acc_history[%d : %d] : %f' % (len(acc_history) // 2, len(acc_history), upper_sum))

    num_centroid_seg = 8
    centroids, seg_len = centroid_segments(acc_history, num_segments=num_centroid_seg)

    # make a plot
    fig, ax = plt.subplots()
    plot_centroid_segments(ax, acc_history, centroids, seg_len)


    #ax.plot(acc_history)
    #xm = single_centroid(acc_history)
    #ax.axvline(x = xm, color='r')

    ax.set_title('Accuracy vs. iterations')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    fig.tight_layout()
    fig.set_size_inches(8, 8)

    plt.show()


if __name__ == '__main__':
    main()
