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
from lernomatic.train import cifar_trainer
from lernomatic.models import cifar
# visualizations
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()

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

def plot_lr_history(ax, lr_history):

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

    model = cifar.CIFAR10Net()
    trainer = cifar_trainer.CIFAR10Trainer(
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
    """
    TestStepLR
    Unit tests for step scheduler object
    """
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
        lr_scheduler = schedule.StepScheduler(
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
        fig1, ax1 = plt.subplots()
        plot_loss_vs_lr(ax1, trainer.loss_history, lr_scheduler.lr_history)
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('step_lr_loss_vs_lr.png', bbox_inches='tight')

        fig2, ax2 = plt.subplots()
        plot_lr_history(ax2, lr_scheduler.lr_history)
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('step_lr_lr_schedule.png', bbox_inches='tight')

        fig3, ax3 = plt.subplots()
        vis_loss_history.plot_train_history(
            ax3,
            trainer.loss_history,
            acc_curve = trainer.acc_history,
            iter_per_epoch = trainer.iter_per_epoch,
            cur_epoch = trainer.cur_epoch
        )
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('step_lr_train_results.png', bbox_inches='tight')

        print('======== TestStepLR.test_train_lr_schedule <END>')


class TestTriangularScheduler(unittest.TestCase):
    """
    TestTriangularScheduler
    Unit tests for the triangular learning rate scheduler
    """
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']

    def test_train_lr_schedule(self):
        print('======== TestTriangularScheduler.test_train_lr_schedule ')

        # get a trainer
        trainer = get_trainer()

        # Get a scheduler. The triangular scheduler will constantly vary
        # between lr_min and lr_max, then back to lr_min over an interval of
        # 2*stepsize.
        test_lr_min = 0.0001
        test_lr_max = 0.01
        test_stepsize = 8 * len(trainer.train_loader)
        test_start_iter = 0
        lr_scheduler = schedule.TriangularScheduler(
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

        fig1, ax1 = plt.subplots()
        plot_loss_vs_lr(ax1, trainer.loss_history, lr_scheduler.lr_history)
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('triangular_lr_loss_vs_lr.png', bbox_inches='tight')

        fig2, ax2 = plt.subplots()
        plot_lr_history(ax2, lr_scheduler.lr_history)
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('triangular_lr_schedule.png', bbox_inches='tight')

        fig3, ax3 = plt.subplots()
        vis_loss_history.plot_train_history(
            ax3,
            trainer.loss_history,
            acc_curve = trainer.acc_history,
            iter_per_epoch = trainer.iter_per_epoch,
            cur_epoch = trainer.cur_epoch
        )
        if GLOBAL_OPTS['draw_plot'] is True:
            plt.show()
        else:
            plt.savefig('triangular_lr_train_results.png', bbox_inches='tight')

        print('======== TestTriangularScheduler.test_train_lr_schedule <END>')


# TODO : warm restart scheduler test


class TestEpochSetScheduler(unittest.TestCase):
    """
    TestEpochSetScheduler
    Unit test for epoch scheduler
    """
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']

    def test_exceptions(self):
        print('======== TestEpochSetScheduler.test_exceptions ')

        with self.assertRaises(ValueError):
            lr_schedule = schedule.EpochSetScheduler(
                [0, 4, 2]
            )

        with self.assertRaises(ValueError):
            lr_schedule = schedule.EpochSetScheduler(
                4.2
            )

        # no zero key, should raise ValueError during check
        epoch_schedule = {
            10: 0.002,
            20: 0.002,
            30: 0.002
        }
        with self.assertRaises(ValueError):
            lr_schedule = schedule.EpochSetScheduler(
                epoch_schedule
            )

        # one of the keys is not an integer
        epoch_schedule = {
            0 : 0.004,
            10: 0.002,
            20: 0.002,
            30: 0.002,
            40.0 : 0.0002
        }
        with self.assertRaises(ValueError):
            lr_schedule = schedule.EpochSetScheduler(
                epoch_schedule
            )
        epoch_schedule = {
            0 : 0.004,
            10: 0.002,
            20: 0.002,
            30: 0.002,
            40: 0.0002
        }

        lr_schedule = schedule.EpochSetScheduler(
            epoch_schedule
        )
        self.assertEqual(False, lr_schedule.lr_value)
        print(lr_schedule)

        print('======== TestEpochSetScheduler.test_exceptions <END>')

    def test_train_lr_schedule(self):
        print('======== TestEpochSetScheduler.test_train_lr_schedule ')

        test_checkpoint_name = 'checkpoint/epoch_set_schedule_train_test.pkl'
        test_save_every = 2000
        # get a trainer
        trainer = get_trainer()
        # get a (valid) schedule
        epoch_schedule = {
            0  : 0.01,
            5  : 0.001,
            20 : 0.0008,
            40 : 0.00008
        }
        train_num_epochs = 50

        lr_schedule = schedule.EpochSetScheduler(
            epoch_schedule,
            lr_value = True
        )
        print(lr_schedule)
        print('Setting schedule in trainer and training')
        trainer.set_lr_scheduler(lr_schedule)
        trainer.set_num_epochs(train_num_epochs)
        trainer.save_every = 0
        trainer.print_every = 200
        trainer.train()

        print('======== TestEpochSetScheduler.test_train_lr_schedule <END>')


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
