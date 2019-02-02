"""
TEST_FIND_LR
Unit test for learning rate finder

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
from lernomatic.vis import vis_loss_history

# debug
from pudb import set_trace; set_trace()

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


class TestLinearFinder(unittest.TestCase):
    def setUp(self):
        self.verbose             = GLOBAL_OPTS['verbose']
        self.test_batch_size     = 32
        self.test_learning_rate  = 0.001
        self.test_lr_num_epochs  = 8            # number of epochs to run test for
        self.test_print_every    = 20
        # options for learning rate finder
        self.test_lr_min         = 1e-4
        self.test_lr_max         = 1e-1
        self.test_num_iter       = 5000
        self.train_num_epochs    = 80

    def get_trainer(self):
        # get a model to test on and its corresponding trainer
        model = cifar10.CIFAR10Net()
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
        print('======== TestLinearFinder.test_find_lr ')

        # get an LRFinder
        trainer = self.get_trainer()
        lr_finder = learning_rate.LinearFinder(
            len(trainer.train_loader),
            lr_min = self.test_lr_min,
            lr_max   = self.test_lr_max,
            num_epochs = self.test_lr_num_epochs,
            verbose  = self.verbose
        )

        if self.verbose:
            print('Created LRFinder object')
            print(lr_finder)

        trainer.find_lr(lr_finder)

        # show plot
        fig1, ax1 = plt.subplots()
        plot_lr_vs_loss(
            ax1,
            lr_finder.log_lr_history,
            lr_finder.smooth_loss_history
        )
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('test_find_lr_lr_vs_loss.png', bbox_inches='tight')

        trainer.print_every = 200
        trainer.train()

        fig2, ax2 = plt.subplots()
        vis_loss_history.plot_loss_history(
            ax2,
            trainer.loss_history,
            acc_curve = trainer.acc_history,
            iter_per_epoch = trainer.iter_per_epoch,
            cur_epoch = trainer.cur_epoch
        )
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('test_find_lr_train_results.png', bbox_inches='tight')


        print('======== TestLinearFinder.test_find_lr <END>')

    #def test_lr_range_find(self):
    #    print('======== TestLinearFinder.test_lr_range_find ')

    #    trainer = self.get_trainer()
    #    lr_finder = learning_rate.LinearFinder(
    #        len(trainer.train_loader),
    #        lr_min = self.test_lr_min,
    #        lr_max   = self.test_lr_max,
    #        num_iter = self.test_num_iter,
    #        num_epochs = self.test_lr_num_epochs,
    #        verbose  = self.verbose
    #    )

    #    # shut linter up
    #    if self.verbose:
    #        print(lr_finder)

    #    trainer.find_lr(lr_finder)
    #    # show plot
    #    fig1, ax1 = plt.subplots()
    #    plot_lr_vs_loss(
    #        ax1,
    #        lr_finder.log_lr_history,
    #        lr_finder.smooth_loss_history
    #    )
    #    if GLOBAL_OPTS['draw_plot'] is True:
    #        plt.show()
    #    else:
    #        plt.savefig('test_lr_range_find_lr_vs_loss.png', bbox_inches='tight')

    #    trainer.print_every = 200
    #    trainer.train()

    #    fig2, ax2 = plt.subplots()
    #    vis_loss_history.plot_loss_history(
    #        ax2,
    #        trainer.loss_history,
    #        acc_curve = trainer.acc_history,
    #        iter_per_epoch = trainer.iter_per_epoch,
    #        cur_epoch = trainer.cur_epoch
    #    )
    #    if GLOBAL_OPTS['draw_plot'] is True:
    #        plt.show()
    #    else:
    #        plt.savefig('test_lr_range_find_train_results.png', bbox_inches='tight')

    #    print('======== TestLinearFinder.test_lr_range_find <END>')


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
