"""
TEST_SEARCHER
Unit test for 'searcher' experiment. This will eventually replace the
LRFinder objects

Stefan Wong 2019
"""

import sys
import unittest
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
# unit(s) under test
from lernomatic.param import learning_rate
from lernomatic.train import cifar10_trainer
from lernomatic.models import cifar10
from lernomatic.models import resnets
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

# helper function for plotting
def plot_lr_find_results(ax, loss_history, smooth_loss_history, lr_history, **kwargs):
    title = kwargs.pop('title', 'Learning rate finder output')

    ax.plot(np.arange(len(loss_history)), loss_history)
    ax.plot(np.arange(len(smooth_loss_history)), smooth_loss_history)
    ax.plot(np.arange(len(lr_history)), lr_history)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss / Learning rate')
    ax.legend(['Loss', 'Smooth Loss', 'Learning rate'])


def plot_lr_vs_loss(ax, lr_history, loss_history):

    ax.plot(lr_history, loss_history)
    ax.set_title('Learning rate vs Loss')
    ax.set_xlabel('(Log) learning rate')
    ax.set_ylabel('(smoothed) loss')


def get_figure():
    fig, ax = plt.subplots()
    return fig, ax


def get_figure_subplots(num_subplots=2):
    fig = plt.figure()
    ax = []
    for p in range(num_subplots):
        sub_ax = fig.add_subplot(num_subplots, 1, (p+1))
        ax.append(sub_ax)

    return fig, ax

# Tests for 'searcher' experiment
class TestSearcher(unittest.TestCase):
    def setUp(self):
        self.verbose     = GLOBAL_OPTS['verbose']
        self.test_lr_min = 3e-6
        self.test_lr_max = 1.0
        self.train_num_epochs = 20
        self.test_num_epochs = 8
        # other trainer params
        self.test_learning_rate = 0.001
        self.test_batch_size = 128
        self.test_print_every = 200

    def get_trainer(self):
        # get a model to test on and its corresponding trainer
        #model = cifar10.CIFAR10Net()
        model = resnets.WideResnet(40, 10)
        trainer = cifar10_trainer.CIFAR10Trainer(
            model,
            # turn off checkpointing
            save_every = 0,
            print_every = self.test_print_every,
            # data options
            batch_size = self.test_batch_size,
            # training options
            learning_rate = self.test_learning_rate,
            num_epochs = self.train_num_epochs,
            device_id = GLOBAL_OPTS['device_id'],
            verbose = self.verbose
        )

        return trainer

    def test_find_lr(self):
        print('======== TestSearcher.test_find_lr ')

        trainer = self.get_trainer()

        lr_finder = learning_rate.LogSearcher(
            trainer,
            explode_thresh = 8,
            num_epochs = self.test_num_epochs,
            lr_min = self.test_lr_min,
            lr_max = self.test_lr_max,
            verbose = True
        )
        print(lr_finder)

        lr_finder.find_lr()

        fig, ax = plt.subplots()
        ax.plot(lr_finder.log_lr_history, lr_finder.smooth_loss_history)
        ax.set_xlabel('lr history (log)')
        ax.set_ylabel('loss history (smooth)')
        ax.set_title('LRSearcher learning rate test')
        fig.savefig('figures/LRSearcher_test.png')

        grad_fig, grad_ax = plt.subplots()
        grad_ax.plot(np.arange(len(lr_finder.loss_grad_history)), lr_finder.loss_grad_history)
        grad_ax.set_xlabel('iteration')
        grad_ax.set_ylabel('loss')
        grad_ax.set_title('Loss gradient')
        grad_fig.savefig('figures/LRSearcher_gradient.png', bbox_inches='tight')


        # now extract the best learning rate range

        print('======== TestSearcher.test_find_lr <END>')



# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--draw-plot',
                        action='store_true',
                        default=False,
                        help='Draw plots'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of worker processes to use for HDF5 load'
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--device-id',
                        type=int,
                        default=0,
                        help='Device to use for tests (default : -1)'
                        )
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    arg_vals = vars(args)
    for k, v in arg_vals.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print('-------- GLOBAL OPTS (%s) --------' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()
