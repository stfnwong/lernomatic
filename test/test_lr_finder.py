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
from lernomatic.param import lr_common
from lernomatic.train import cifar_trainer
from lernomatic.models import cifar
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

# helper function for plotting
def plot_lr_find_results(ax,
                         loss_history:np.ndarray,
                         smooth_loss_history:np.ndarray,
                         lr_history:np.ndarray,
                         title:str='Learning rate finder output') -> None:
    ax.plot(np.arange(len(loss_history)), loss_history)
    ax.plot(np.arange(len(smooth_loss_history)), smooth_loss_history)
    ax.plot(np.arange(len(lr_history)), lr_history)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss / Learning rate')
    ax.legend(['Loss', 'Smooth Loss', 'Learning rate'])


def plot_lr_vs_loss(ax, lr_history:np.ndarray, loss_history:np.ndarray) -> None:

    ax.plot(lr_history, loss_history)
    ax.set_title('Learning rate vs Loss')
    ax.set_xlabel('(Log) learning rate')
    ax.set_ylabel('(smoothed) loss')


def get_figure() -> tuple:
    fig, ax = plt.subplots()
    return (fig, ax)


GLOBAL_TEST_PARAMS = {
        'test_batch_size'        : 32,
        'test_learning_rate'     : 0.001,
        'test_lr_num_epochs'     : 4,            # number of epochs to run test for
        'test_print_every'       : 20,
        # options for learning rate finder
        'test_lr_min'            : 1e-8,
        'test_lr_max'            : 1.0,
        'test_num_iter'          : 5000,
        'test_lr_explode_thresh' : 4.5,
        'train_num_epochs'       : 80,
}


# Helper function to generate a trainer object
def get_trainer() -> cifar_trainer.CIFAR10Trainer:
    # get a model to test on and its corresponding trainer
    model = cifar.CIFAR10Net()
    trainer = cifar_trainer.CIFAR10Trainer(
        model,
        # turn off checkpointing
        save_every = 0,
        print_every = GLOBAL_TEST_PARAMS['test_print_every'],
        # data options
        batch_size = GLOBAL_TEST_PARAMS['test_batch_size'],
        # training options
        learning_rate = GLOBAL_TEST_PARAMS['test_learning_rate'],
        num_epochs = GLOBAL_TEST_PARAMS['train_num_epochs'],
        device_id = GLOBAL_OPTS['device_id'],
        verbose = GLOBAL_OPTS['verbose']
    )

    return trainer


class TestLogFinder(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']

    def test_find_lr(self):
        print('======== TestLogFinder.test_find_lr ')

        # get an LRFinder
        trainer = get_trainer()
        lr_finder = lr_common.LogFinder(
            trainer,
            lr_min         = GLOBAL_TEST_PARAMS['test_lr_min'],
            lr_max         = GLOBAL_TEST_PARAMS['test_lr_max'],
            num_epochs     = GLOBAL_TEST_PARAMS['test_lr_num_epochs'],
            explode_thresh = GLOBAL_TEST_PARAMS['test_lr_explode_thresh'],
            verbose        = GLOBAL_OPTS['verbose']
        )

        if self.verbose:
            print('Created LRFinder object')
            print(lr_finder)

        lr_find_min, lr_find_max = lr_finder.find()
        print('Found learning rate range as %.3f -> %.3f' % (lr_find_min, lr_find_max))

        # show plot
        finder_fig, finder_ax = vis_loss_history.get_figure_subplots(2)
        lr_finder.plot_lr_vs_acc(finder_ax[0])
        lr_finder.plot_lr_vs_loss(finder_ax[1])
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('figures/test_find_lr_plots.png', bbox_inches='tight')

        # train the network with the discovered parameters
        trainer.print_every = 200
        trainer.train()

        train_fig, train_ax = vis_loss_history.get_figure_subplots()
        vis_loss_history.plot_train_history_2subplots(
            train_ax,
            trainer.get_loss_history(),
            acc_curve = trainer.get_acc_history(),
            iter_per_epoch = trainer.iter_per_epoch,
            cur_epoch = trainer.cur_epoch
        )
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            train_fig.savefig('figures/test_find_lr_train_results.png', bbox_inches='tight')

        print('======== TestLogFinder.test_find_lr <END>')

    def test_lr_range_find(self):
        print('======== TestLogFinder.test_lr_range_find ')

        trainer = get_trainer()
        lr_finder = lr_common.LogFinder(
            trainer,
            lr_min     = GLOBAL_TEST_PARAMS['test_lr_min'],
            lr_max     = GLOBAL_TEST_PARAMS['test_lr_max'],
            num_iter   = GLOBAL_TEST_PARAMS['test_num_iter'],
            num_epochs = GLOBAL_TEST_PARAMS['test_lr_num_epochs'],
            acc_test   = True,
            verbose    = GLOBAL_OPTS['verbose']
        )

        # shut linter up
        if self.verbose:
            print(lr_finder)

        lr_find_min, lr_find_max = lr_finder.find()
        # show plot
        fig1, ax1 = plt.subplots()
        lr_finder.plot_lr_vs_acc(ax1)

        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('figures/test_lr_range_find_lr_vs_acc.png', bbox_inches='tight')

        trainer.print_every = 200
        trainer.train()

        fig2, ax2 = vis_loss_history.get_figure_subplots()
        vis_loss_history.plot_train_history_2subplots(
            ax2,
            trainer.get_loss_history(),
            acc_curve = trainer.get_acc_history(),
            iter_per_epoch = trainer.iter_per_epoch,
            cur_epoch = trainer.cur_epoch
        )
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('figures/test_lr_range_find_train_results.png', bbox_inches='tight')

        print('======== TestLogFinder.test_lr_range_find <END>')

    def test_model_param_save(self):c
        print('======== TestLogFinder.test_model_param_save ')

        # get a trainer, etcC:w
        trainer = get_trainer()
        lr_finder = lr_common.LogFinder(
            trainer,
            lr_min     = GLOBAL_TEST_PARAMS['test_lr_min'],
            lr_max     = GLOBAL_TEST_PARAMS['test_lr_max'],
            num_iter   = GLOBAL_TEST_PARAMS['test_num_iter'],
            num_epochs = GLOBAL_TEST_PARAMS['test_lr_num_epochs'],
            acc_test   = True,
            verbose    = GLOBAL_OPTS['verbose']
        )

        # shut linter up
        if self.verbose:
            print(lr_finder)

        # make a copy of the model parameters before we start looking for a new
        # learning rate.
        lr_find_min, lr_find_max = lr_finder.find()
        # show plot
        fig1, ax1 = plt.subplots()
        lr_finder.plot_lr_vs_acc(ax1)

        # now check that the restored parameters match the copy of the
        # parameters save earlier

        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('figures/test_lr_range_find_lr_vs_acc.png', bbox_inches='tight')

        trainer.print_every = 200
        trainer.train()

        fig2, ax2 = vis_loss_history.get_figure_subplots()
        vis_loss_history.plot_train_history_2subplots(
            ax2,
            trainer.get_loss_history(),
            acc_curve = trainer.get_acc_history(),
            iter_per_epoch = trainer.iter_per_epoch,
            cur_epoch = trainer.cur_epoch
        )
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('figures/test_lr_range_find_train_results.png', bbox_inches='tight')


        print('======== TestLogFinder.test_model_param_save <END>')


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
            print('\t[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()
