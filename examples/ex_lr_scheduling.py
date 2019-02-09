"""
EX_LR_SCHEDULING
Example showing the various learning rate scheduling options

Stefan Wong 2019
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from lernomatic.param import learning_rate
from lernomatic.vis import vis_lr
# we use CIFAR-10 for this example
from lernomatic.models import cifar10
from lernomatic.train import cifar10_trainer
from lernomatic.train import schedule
# vis tools
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def triangular_sched():
    # get a model and trainer
    triangular_sched_model = cifar10.CIFAR10Net()
    #model = resnets.WideResnet(28, 10)
    triangular_sched_trainer = cifar10_trainer.CIFAR10Trainer(
        triangular_sched_model,
        batch_size      = GLOBAL_OPTS['batch_size'],
        test_batch_size = GLOBAL_OPTS['test_batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        #momentum = GLOBAL_OPTS['momentum'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = 'triangular_schedule_cifar10',
        # display,
        print_every     = GLOBAL_OPTS['print_every'],
        save_every      = GLOBAL_OPTS['save_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    # get an LRFinder object
    lr_finder = learning_rate.LogFinder(
        triangular_sched_trainer,
        lr_min         = GLOBAL_OPTS['lr_min'],
        lr_max         = GLOBAL_OPTS['lr_max'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = GLOBAL_OPTS['find_print_every']
    )

    lr_finder.find()        # TODO: still need automatic lr range setting

    lr_find_max = 1e-1
    lr_find_min = 1e-2

    lr_scheduler = schedule.TriangularScheduler(
        stepsize = int(len(triangular_sched_trainer.train_loader) / 4),
        lr_min = lr_find_min,
        lr_max = lr_find_max
    )

    triangular_sched_trainer.set_lr_scheduler(lr_scheduler)
    triangular_sched_trainer.train()

    # generate loss history plot
    train_fig, train_ax = vis_loss_history.get_figure_subplots()
    vis_loss_history.plot_train_history_2subplots(
        train_ax,
        triangular_sched_trainer.loss_history,
        acc_history = triangular_sched_trainer.acc_history[0 : triangular_sched_trainer.acc_iter],
        cur_epoch = triangular_sched_trainer.cur_epoch,
        iter_per_epoch = triangular_sched_trainer.iter_per_epoch,
        loss_title = 'CIFAR-10 LR Finder Loss\n (%s min LR: %f, max LR: %f' % (repr(lr_scheduler), lr_scheduler.lr_min, lr_scheduler.lr_max),
        acc_title = 'CIFAR-10 LR Finder Accuracy '
    )
    train_fig.savefig('figures/ex_triangular_sched_cifar10.png', bbox_inches='tight')


def triangular2_sched():
    # get a model and trainer
    triangular2_sched_model = cifar10.CIFAR10Net()
    #model = resnets.WideResnet(28, 10)
    triangular2_sched_trainer = cifar10_trainer.CIFAR10Trainer(
        triangular2_sched_model,
        batch_size      = GLOBAL_OPTS['batch_size'],
        test_batch_size = GLOBAL_OPTS['test_batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        #momentum = GLOBAL_OPTS['momentum'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = 'triangular2_schedule_cifar10',
        # display,
        print_every     = GLOBAL_OPTS['print_every'],
        save_every      = GLOBAL_OPTS['save_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    # get an LRFinder object
    lr_finder = learning_rate.LogFinder(
        triangular2_sched_trainer,
        lr_min         = GLOBAL_OPTS['lr_min'],
        lr_max         = GLOBAL_OPTS['lr_max'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = GLOBAL_OPTS['find_print_every']
    )

    lr_finder.find()        # TODO: still need automatic lr range setting

    lr_find_max = 1e-1
    lr_find_min = 1e-2

    lr_scheduler = schedule.Triangular2Scheduler(
        stepsize = int(len(triangular2_sched_trainer.train_loader) / 4),
        lr_min = lr_find_min,
        lr_max = lr_find_max
    )

    triangular2_sched_trainer.set_lr_scheduler(lr_scheduler)
    triangular2_sched_trainer.train()

    # generate loss history plot
    train_fig, train_ax = vis_loss_history.get_figure_subplots()
    vis_loss_history.plot_train_history_2subplots(
        train_ax,
        triangular2_sched_trainer.loss_history,
        acc_history = triangular2_sched_trainer.acc_history[0 : triangular2_sched_trainer.acc_iter],
        cur_epoch = triangular2_sched_trainer.cur_epoch,
        iter_per_epoch = triangular2_sched_trainer.iter_per_epoch,
        loss_title = 'CIFAR-10 LR Finder Loss\n (%s min LR: %f, max LR: %f' % (repr(lr_scheduler), lr_scheduler.lr_min, lr_scheduler.lr_max),
        acc_title = 'CIFAR-10 LR Finder Accuracy '
    )
    train_fig.savefig('figures/ex_triangular2_sched_cifar10.png', bbox_inches='tight')


def step_sched():
    # get a model and trainer
    step_sched_model = cifar10.CIFAR10Net()
    #model = resnets.WideResnet(28, 10)
    step_sched_trainer = cifar10_trainer.CIFAR10Trainer(
        step_sched_model,
        batch_size      = GLOBAL_OPTS['batch_size'],
        test_batch_size = GLOBAL_OPTS['test_batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        #momentum = GLOBAL_OPTS['momentum'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = 'step_schedule_cifar10',
        # display,
        print_every     = GLOBAL_OPTS['print_every'],
        save_every      = GLOBAL_OPTS['save_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    # get an LRFinder object
    lr_finder = learning_rate.LogFinder(
        step_sched_trainer,
        lr_min         = GLOBAL_OPTS['lr_min'],
        lr_max         = GLOBAL_OPTS['lr_max'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = GLOBAL_OPTS['find_print_every']
    )

    lr_finder.find()        # TODO: still need automatic lr range setting

    lr_find_max = 1e-1
    lr_find_min = 1e-2

    lr_scheduler = schedule.StepScheduler(
        stepsize = int(len(step_sched_trainer.train_loader) / 4),
        lr_min = lr_find_min,
        lr_max = lr_find_max,
        lr_decay_every = int(len(step_sched_trainer.train_loader) / 4),
        lr_decay = 0.001
    )

    step_sched_trainer.set_lr_scheduler(lr_scheduler)
    step_sched_trainer.train()

    # generate loss history plot
    train_fig, train_ax = vis_loss_history.get_figure_subplots()
    vis_loss_history.plot_train_history_2subplots(
        train_ax,
        step_sched_trainer.loss_history,
        acc_history = step_sched_trainer.acc_history[0 : step_sched_trainer.acc_iter],
        cur_epoch = step_sched_trainer.cur_epoch,
        iter_per_epoch = step_sched_trainer.iter_per_epoch,
        loss_title = 'CIFAR-10 LR Finder Loss\n (%s min LR: %f, max LR: %f' % (repr(lr_scheduler), lr_scheduler.lr_min, lr_scheduler.lr_max),
        acc_title = 'CIFAR-10 LR Finder Accuracy '
    )
    train_fig.savefig('figures/ex_step_sched_cifar10.png', bbox_inches='tight')


def warm_restart_sched():
    warm_restart_model = cifar10.CIFAR10Net()
    #model = resnets.WideResnet(28, 10)
    warm_restart_trainer = cifar10_trainer.CIFAR10Trainer(
        warm_restart_model,
        batch_size      = GLOBAL_OPTS['batch_size'],
        test_batch_size = GLOBAL_OPTS['test_batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        #momentum = GLOBAL_OPTS['momentum'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = 'step_schedule_cifar10',
        # display,
        print_every     = GLOBAL_OPTS['print_every'],
        save_every      = GLOBAL_OPTS['save_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    # get an LRFinder object
    lr_finder = learning_rate.LogFinder(
        warm_restart_trainer,
        lr_min         = GLOBAL_OPTS['lr_min'],
        lr_max         = GLOBAL_OPTS['lr_max'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = GLOBAL_OPTS['find_print_every']
    )

    lr_finder.find()        # TODO: still need automatic lr range setting

    lr_find_max = 1e-1
    lr_find_min = 1e-2

    lr_scheduler = schedule.WarmRestartScheduler(
        stepsize = int(len(warm_restart_trainer.train_loader) / 4),
        lr_min = lr_find_min,
        lr_max = lr_find_max,
        lr_decay_every = int(len(warm_restart_trainer.train_loader) / 4),
        lr_decay = 0.001
    )

    warm_restart_trainer.set_lr_scheduler(lr_scheduler)
    warm_restart_trainer.train()

    # generate loss history plot
    train_fig, train_ax = vis_loss_history.get_figure_subplots()
    vis_loss_history.plot_train_history_2subplots(
        train_ax,
        warm_restart_trainer.loss_history,
        acc_history = warm_restart_trainer.acc_history[0 : warm_restart_trainer.acc_iter],
        cur_epoch = warm_restart_trainer.cur_epoch,
        iter_per_epoch = warm_restart_trainer.iter_per_epoch,
        loss_title = 'CIFAR-10 LR Finder Loss\n (%s min LR: %f, max LR: %f' % (repr(lr_scheduler), lr_scheduler.lr_min, lr_scheduler.lr_max),
        acc_title = 'CIFAR-10 LR Finder Accuracy '
    )
    train_fig.savefig('figures/ex_warm_restart_sched_cifar10.png', bbox_inches='tight')


def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--draw-plot',
                        default=False,
                        action='store_true',
                        help='Display plots'
                        )
    parser.add_argument('--print-every',
                        type=int,
                        default=100,
                        help='Print output every N epochs'
                        )
    parser.add_argument('--save-every',
                        type=int,
                        default=1000,
                        help='Save model checkpoint every N epochs'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of workers to use when generating HDF5 files'
                        )
    # Learning rate finder options
    parser.add_argument('--find-print-every',
                        type=int,
                        default=20,
                        help='How often to print output from learning rate finder'
                        )
    parser.add_argument('--find-num-epochs',
                        type=int,
                        default=8,
                        help='Maximum number of epochs to attempt to find learning rate'
                        )
    parser.add_argument('--find-explode-thresh',
                        type=float,
                        default=4.5,
                        help='Threshold at which to stop increasing learning rate'
                        )
    parser.add_argument('--lr-min',
                        type=float,
                        default=2e-4,
                        help='Minimum range to search for learning rate'
                        )
    parser.add_argument('--lr-max',
                        type=float,
                        default=1e-1,
                        help='Maximum range to search for learning rate'
                        )
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )
    # Network options
    # Training options
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during testing'
                        )
    parser.add_argument('--start-epoch',
                        type=int,
                        default=0,
                        )
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Epoch to stop training at'
                        )

    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.0,
                        help='Weight decay to use for optimizer'
                        )
    parser.add_argument('--learning-rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate for optimizer'
                        )
    # Data options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='lr_find_ex_cifar10',
                        help='Name to prepend to all checkpoints'
                        )
    parser.add_argument('--load-checkpoint',
                        type=str,
                        default=None,
                        help='Load a given checkpoint'
                        )
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing processed data files'
                        )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS ---- ')
        for k,v in GLOBAL_OPTS.items():
            print('%s : %s' % (str(k), str(v)))

    # Execute each of the trainers in turn
    triangular_sched()
    triangular2_sched()
    step_sched()
    #warm_restart_sched()
