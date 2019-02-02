"""
TEST_SCHEDULER
Units tests for Scheduler objects

Stefan Wong 2019
"""

import sys
import unittest
import argparse
import numpy as np
import matplotlib.pyplot as plt

# modules under test
from lernomatic.train import schedule
from lernomatic.train import cifar10_trainer
from lernomatic.models import cifar10

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

# helper functions for plotting
def plot_loss_vs_lr(ax, loss_history, lr_history):
    if len(lr_history) != len(loss_history):
        raise ValueError('lr_history and loss_history must be same length')

    #ax.plot(loss_history, lr_history)
    ax.plot(np.arange(len(loss_history)), loss_history)
    ax.plot(np.arange(len(lr_history)), lr_history)
    ax.set_xlabel('Loss')
    ax.set_ylabel('Learning Rate')
    ax.set_title('loss vs learning rate')
    ax.legend(['loss history', 'lr history'])

def plot_lr_schedule(ax, lr_history):

    ax.plot(np.arange(len(lr_history)), lr_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning rate')
    ax.set_title('Final learning rate schedule')


# helper function to get trainer and model
def get_trainer(learning_rate = None):

    if learning_rate is not None:
        test_learning_rate = learning_rate
    else:
        test_learning_rate = GLOBAL_OPTS['learning_rate']

    model = cifar10.CIFAR10Net()
    trainer = cifar10_trainer.CIFAR10Trainer(
        model,
        # data options
        batch_size = GLOBAL_OPTS['batch_size'],
        num_workers = GLOBAL_OPTS['num_workers'],
        num_epochs = GLOBAL_OPTS['num_epochs'],
        # set initial learning rate
        learning_rate = test_learning_rate,
        # other options
        save_every = 0,
        print_every = 200,
        device_id = GLOBAL_OPTS['device_id'],
        verbose = GLOBAL_OPTS['verbose']
    )

    return trainer


class TestStepLR(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']

    def test_train_lr_schedule(self):
        print('======== TestStepLR.test_train_lr_schedule ')

        # Create some parameters for the test. Since this is step annealing
        # we will be varying the learning rate starting at lr_max, and
        # decreasing towards lr_min by a factor of test_lr_decay every 10000
        # iterations
        test_lr_min = 1e-4
        test_lr_max = 1e-1
        test_lr_decay = 0.01
        test_lr_decay_every = 1000
        test_start_iter = 0

        # get a trainer
        trainer = get_trainer()
        # get a scheduler
        lr_scheduler = schedule.StepLRScheduler(
            lr_min = test_lr_min,
            lr_max = test_lr_max,
            lr_decay = test_lr_decay,
            lr_decay_every = test_lr_decay_every,
            start_iter = test_start_iter,
            lr_history_size = len(trainer.loss_history)
        )
        if self.verbose:
            print(lr_scheduler)
        trainer.set_lr_scheduler(lr_scheduler)
        trainer.train()

        print('Generating loss history vs learning rate history plot')
        fig, ax = plt.subplots()
        plot_loss_vs_lr(ax, trainer.loss_history, lr_scheduler.lr_history)
        plt.show()
        plot_lr_schedule(ax, lr_scheduler.lr_history)
        plt.show()

        print('======== TestStepLR.test_train_lr_schedule <END>')


class TestTriangularLR(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']

    def test_train_lr_schedule(self):
        print('======== TestTriangularLR.test_train_lr_schedule ')

        # get a trainer
        trainer = get_trainer()

        # Get a scheduler. The triangular scheduler will constantly vary
        # between lr_min and lr_max, then back to lr_min over an interval of
        # 2*stepsize.
        test_lr_min = 0.0001
        test_lr_max = 0.01
        test_stepsize = 8 * len(trainer.train_loader)
        test_start_iter = 0
        lr_scheduler = schedule.TriangularLRScheduler(
            lr_min = test_lr_min,
            lr_max = test_lr_max,
            stepsize = test_stepsize,
            start_iter = test_start_iter,
            lr_history_size = len(trainer.loss_history)
        )
        if self.verbose:
            print(lr_scheduler)
        trainer.set_lr_scheduler(lr_scheduler)
        trainer.train()

        print('Generating loss history vs learning rate history plot')
        fig, ax = plt.subplots()
        plot_loss_vs_lr(ax, trainer.loss_history, lr_scheduler.lr_history)
        plt.show()
        plot_lr_schedule(ax, lr_scheduler.lr_history)
        plt.show()

        print('======== TestTriangularLR.test_train_lr_schedule <END>')

    # TODO : test where we use the learning rate finder to pick the bounds

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
                        default=32,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--learning-rate',
                        type=float,
                        default=0.0001,
                        help='Initial learning rate to use for test'
                        )
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Number of epochs to train for in test'
                        )
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
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
