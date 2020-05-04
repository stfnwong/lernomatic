"""
EX_LR_SCHEDULING
Example showing the various learning rate scheduling options

Stefan Wong 2019
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from lernomatic.param import lr_common
from lernomatic.vis import vis_lr
# we use CIFAR-10 for this example
from lernomatic.models import common
from lernomatic.models import cifar
from lernomatic.models import resnets
from lernomatic.train import trainer
from lernomatic.train import cifar_trainer
from lernomatic.train import schedule
# vis tools
from lernomatic.vis import vis_loss_history

# debug
#

GLOBAL_OPTS = dict()


# Helper functions for models
def get_model() -> common.LernomaticModel:
    if GLOBAL_OPTS['model'] == 'resnet':
        model = resnets.WideResnet(
            depth=GLOBAL_OPTS['resnet_depth'],
            num_classes = 10,
            input_channels = 3
        )
    elif GLOBAL_OPTS['model'] == 'cifar':
        model = cifar.CIFAR10Net()
    else:
        raise ValueError('Unknown model type [%s]' % str(GLOBAL_OPTS['model']))

    return model


# Helper function for finder
def get_lr_finder(trainer, find_type='LogFinder') -> lr_common.LRFinder:
    if not hasattr(lr_common, find_type):
        raise ValueError('Unknown learning rate finder type [%s]' % str(find_type))

    lr_find_obj = getattr(lr_common, find_type)
    lr_finder = lr_find_obj(
        trainer,
        lr_min         = GLOBAL_OPTS['lr_min'],
        lr_max         = GLOBAL_OPTS['lr_max'],
        lr_select_method = GLOBAL_OPTS['lr_select_method'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = GLOBAL_OPTS['find_print_every']
    )

    return lr_finder


# Helper function for scheduler
def get_scheduler(lr_min:float,
                  lr_max:float,
                  stepsize:int,
                  sched_type:str='TriangularScheduler') -> schedule.LRScheduler:
    if sched_type is None:
        return None

    if not hasattr(schedule, sched_type):
        raise ValueError('Unknown scheduler type [%s]' % str(sched_type))

    lr_sched_obj = getattr(schedule, sched_type)
    lr_scheduler = lr_sched_obj(
        lr_min = lr_min,
        lr_max = lr_max,
        stepsize = stepsize
    )

    return lr_scheduler


# Helper function for trainer
def get_trainer(model, checkpoint_name):
    trainer = cifar_trainer.CIFAR10Trainer(
        model,
        batch_size      = GLOBAL_OPTS['batch_size'],
        val_batch_size  = GLOBAL_OPTS['val_batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        #momentum = GLOBAL_OPTS['momentum'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = checkpoint_name,
        # display,
        print_every     = GLOBAL_OPTS['print_every'],
        save_every      = GLOBAL_OPTS['save_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    return trainer


# perform the required schedule
def run_schedule(trainer, sched_type, checkpoint_name, lr_min=None, lr_max=None):
    # get model and trainer

    # get finder
    # NOTE : the finder doesn't need to be re-run each time. While we do need a
    # new trainer object (to have seperate history files for each run) the way
    # the example is set up we could just run the lr_finder.find() routine once
    # and then re-use the same ranges for each schedule. TODO : make this an
    # option

    if (lr_min is None) or (lr_max is None):
        lr_finder = get_lr_finder(trainer)
        lr_finder.find()
        lr_find_min, lr_find_max = lr_finder.get_lr_range()
        print('Found learning rate range %.4f -> %.4f' % (lr_find_min, lr_find_max))

    if GLOBAL_OPTS['find_only'] is True:
        return lr_finder

    stepsize = len(trainer.train_loader)
    # get scheduler
    lr_scheduler = get_scheduler(
        lr_min,
        lr_max,
        stepsize,
        sched_type
    )
    print('Got scheduler [%s]' % repr(lr_scheduler))
    # train
    trainer.set_lr_scheduler(lr_scheduler)
    trainer.train()

    # TODO : determine mutability of trainer object
    return trainer


# Find lr for a given trainer
def find_lr(trainer, return_finder=False):
    lr_finder = get_lr_finder(trainer)
    lr_finder.find()
    lr_find_min, lr_find_max = lr_finder.get_lr_range()
    print('Found learning rate range %.4f -> %.4f' % (lr_find_min, lr_find_max))

    if return_finder is True:
        return (lr_find_min, lr_find_max, lr_finder)
    else:
        return (lr_find_min, lr_find_max)


#  create plot to save to disk
def generate_plot(trainer, loss_title, acc_title, fig_filename):
    train_fig, train_ax = vis_loss_history.get_figure_subplots(2)
    vis_loss_history.plot_train_history_2subplots(
        train_ax,
        trainer.get_loss_history(),
        acc_history = trainer.get_acc_history(),
        cur_epoch = trainer.cur_epoch,
        iter_per_epoch = trainer.iter_per_epoch,
        loss_title = loss_title,
        acc_title = acc_title
    )
    train_fig.tight_layout()
    train_fig.savefig(fig_filename)



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
                        default=-1,
                        help='Save model checkpoint every N epochs'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of workers to use when generating HDF5 files'
                        )
    parser.add_argument('--find-only',
                        action='store_true',
                        default=False,
                        help='Only perform the parameter find step (no training)'
                        )
    # model, size options
    parser.add_argument('--model',
                        type=str,
                        default='resnet',
                        help='Type of model to use in example. (default: resnet)'
                        )
    parser.add_argument('--resnet-depth',
                        type=int,
                        default=58,
                        help='Depth of resnet to use for resnet models'
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
    parser.add_argument('--lr-select-method',
                        type=str,
                        default='min_loss',
                        help='Method to use for selecting LR range'
                        )
    # Schedule options
    parser.add_argument('--exp-decay',
                        type=float,
                        default=0.001,
                        help='Exponential decay term'
                        )
    parser.add_argument('--sched-stepsize',
                        type=int,
                        default=4000,
                        help='Size of step for learning rate scheduler'
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
    parser.add_argument('--val-batch-size',
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
            print('\t[%s] : %s' % (str(k), str(v)))

    schedulers = [
        'TriangularScheduler',
        'Triangular2Scheduler',
        'ExponentialDecayScheduler',
        'WarmRestartScheduler',
        'TriangularExpScheduler',
        'Triangular2ExpScheduler',
        'DecayWhenAcc',
        'TriangularDecayWhenAcc',
        None
    ]

    checkpoint_names = [
        str(GLOBAL_OPTS['model']) + '_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_triangular_sched_cifar10',
        str(GLOBAL_OPTS['model']) + '_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_triangular2_sched_cifar10',
        str(GLOBAL_OPTS['model']) + '_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_exp_decay_sched_cifar10',
        str(GLOBAL_OPTS['model']) + '_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_warm_restart_sched_cifar10',
        str(GLOBAL_OPTS['model']) + '_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_triangular_exp_sched_cifar10',
        str(GLOBAL_OPTS['model']) + '_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_triangular2_exp_sched_cifar10',
        str(GLOBAL_OPTS['model']) + '_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_decay_when_acc_cifar10',
        str(GLOBAL_OPTS['model']) + '_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_triangular_decay_when_acc_cifar10',
        str(GLOBAL_OPTS['model']) + '_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_no_sched_cifar10'
    ]

    figure_dir = 'figures/'
    figure_names = [
        figure_dir + '[' + str(GLOBAL_OPTS['model']) + ']_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_ex_triangular_sched_cifar10.png',
        figure_dir + '[' + str(GLOBAL_OPTS['model']) + ']_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_ex_triangular2_sched_cifar10.png',
        figure_dir + '[' + str(GLOBAL_OPTS['model']) + ']_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_ex_exp_decay_sched_cifar10.png',
        figure_dir + '[' + str(GLOBAL_OPTS['model']) + ']_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_ex_warm_restart_sched_cifar10.png',
        figure_dir + '[' + str(GLOBAL_OPTS['model']) + ']_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_ex_triangular_exp_sched_cifar10.png',
        figure_dir + '[' + str(GLOBAL_OPTS['model']) + ']_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_ex_triangular2_exp_sched_cifar10.png',
        figure_dir + '[' + str(GLOBAL_OPTS['model']) + ']_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_ex_decay_when_acc_cifar10.png',
        figure_dir + '[' + str(GLOBAL_OPTS['model']) + ']_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_ex_triangular_decay_when_acc_cifar10.png',
        figure_dir + '[' + str(GLOBAL_OPTS['model']) + ']_[' + str(GLOBAL_OPTS['lr_select_method']) + ']_ex_no_sched_cifar10.png',
    ]

    assert len(schedulers) == len(checkpoint_names)
    assert len(schedulers) == len(figure_names)

    trainers = []
    acc_list = []
    # run each schedule
    for idx in range(len(schedulers)):

        model = get_model()
        trainer = get_trainer(
            model,
            checkpoint_names[idx]
        )

        if idx == 0:
            lr_find_min, lr_find_max, lr_finder = find_lr(trainer, return_finder=True)
            # create plots
            lr_acc_title  = '[' + str(GLOBAL_OPTS['model']) + '[' + str(GLOBAL_OPTS['lr_select_method']) + '] ' +\
                str(schedulers[idx]) + ' learning rate vs acc (log)'
            lr_loss_title = '[' + str(GLOBAL_OPTS['model']) + '[' + str(GLOBAL_OPTS['lr_select_method']) + '] ' +\
                str(schedulers[idx]) + ' learning rate vs loss (log)'
            lr_fig, lr_ax = vis_loss_history.get_figure_subplots(2)
            lr_finder.plot_lr_vs_acc(lr_ax[0], lr_acc_title, log=True)
            lr_finder.plot_lr_vs_loss(lr_ax[1], lr_loss_title, log=True)
            # save
            lr_fig.tight_layout()
            lr_fig.savefig('figures/[%s][%s]_%s_lr_finder_output.png' %\
                           (str(GLOBAL_OPTS['model']), str(GLOBAL_OPTS['lr_select_method']), str(schedulers[idx]))
            )

        print('Found learning rates as %.4f -> %.4f' % (lr_find_min, lr_find_max))
        if GLOBAL_OPTS['find_only'] is True:
            break

        run_schedule(
            trainer,
            schedulers[idx],
            checkpoint_names[idx],
            lr_min = lr_find_min,
            lr_max = lr_find_max
        )
        # create plots
        trainers.append(trainer)
        acc_list.append(trainer.get_acc_history())
        loss_title = str(schedulers[idx]) + ' ' + str(GLOBAL_OPTS['lr_select_method']) + ' Loss : LR range (%.3f -> %.3f)' % (lr_find_min, lr_find_max)
        acc_title  = str(schedulers[idx]) + ' ' + str(GLOBAL_OPTS['lr_select_method']) + ' Accuracy : LR range (%.3f -> %.3f)' % (lr_find_min, lr_find_max)
        generate_plot(trainer, loss_title, acc_title, figure_names[idx])


    # Make one more plot of comparing accuracies
    if GLOBAL_OPTS['find_only'] is False:
        acc_fig, acc_ax = plt.subplots()
        for acc in acc_list:
            acc_ax.plot(np.arange(len(acc)), acc)
        acc_ax.set_xlabel('Epoch')
        acc_ax.set_ylabel('Accuracy')
        acc_ax.legend(checkpoint_names)
        acc_ax.set_title('[%s] [%s] Accuracy comparison for learning rate schedules (LR: %.4f -> %.4f)' %\
                        (str(GLOBAL_OPTS['model']), str(GLOBAL_OPTS['lr_select_method']), lr_find_min, lr_find_max)
        )
        acc_fig.tight_layout()
        acc_fig.set_size_inches(10, 10)
        acc_fig.savefig('figures/[%s]_[%s]_ex_lr_scheduling_acc_compare_%.4f_%.4f.png' %\
                        (str(GLOBAL_OPTS['model']), str(GLOBAL_OPTS['lr_select_method']), lr_find_min, lr_find_max)
        )
