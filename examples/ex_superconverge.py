"""
EX_SUPERCONVERGE
Example implementing super-convergence.

Stefan Wong 2019
"""

# TODO : extend this to also use imagenet, COCO, etc

import time
from datetime import timedelta
import argparse
from lernomatic.models import common
from lernomatic.models import cifar
from lernomatic.models import resnets
from lernomatic.train import trainer
from lernomatic.train import cifar_trainer
from lernomatic.train import schedule
from lernomatic.param import lr_common
from lernomatic.vis import vis_loss_history


# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def get_model(model_type:str,
              num_classes:int=10,
              input_channels:int=3,
              depth:int=58) -> common.LernomaticModel:
    """
    Get new model objects
    """
    if model_type == 'resnet':
        model = resnets.WideResnet(
            depth=depth,
            num_classes=num_classes,
            input_channels = input_channels
        )
    elif model_type == 'cifar':
        model = cifar.CIFAR10Net()
    else:
        raise ValueError('Unknown model type [%s]' % str(GLOBAL_OPTS['model']))

    return model


def get_lr_finder(trainer:trainer.Trainer,
                  lr_min:float,
                  lr_max:float,
                  lr_select_method:str='max_acc',
                  find_num_epochs:int=4,
                  explode_thresh:float=8.0,
                  print_every:int=100,
                  max_batches:int=0,
                  find_type:str='LogFinder') -> lr_common.LRFinder:
    """
    Get new learning rate finder objects
    """
    if not hasattr(lr_common, find_type):
        raise ValueError('Unknown learning rate finder type [%s]' % str(find_type))

    lr_find_obj = getattr(lr_common, find_type)
    lr_finder = lr_find_obj(
        trainer,
        lr_min           = lr_min,
        lr_max           = lr_max,
        lr_select_method = lr_select_method,
        num_epochs       = find_num_epochs,
        explode_thresh   = explode_thresh,
        max_batches      = max_batches,
        print_every      = print_every
    )

    return lr_finder


def get_scheduler(lr_min:float,
                  lr_max:float,
                  stepsize:int,
                  sched_type:str='TriangularScheduler') -> schedule.LRScheduler:
    """
    Get new scheduler objects
    """
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


def get_trainer(model:common.LernomaticModel,
                checkpoint_name:str,
                trainer_type:str='cifar') -> trainer.Trainer:
    """
    Get new trainer objects
    """
    if trainer_type == 'cifar':
        t = cifar_trainer.CIFAR10Trainer(
            model,
            # initial learning rate
            learning_rate   = GLOBAL_OPTS['learning_rate'],
            num_epochs      = GLOBAL_OPTS['num_epochs'],
            batch_size      = GLOBAL_OPTS['batch_size'],
            stop_when_acc   = GLOBAL_OPTS['stop_when_acc'],
            # optimization
            optim_function  = 'SGD',
            # device
            device_id       = GLOBAL_OPTS['device_id'],
            # checkpoint
            checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
            checkpoint_name = checkpoint_name,
            # other
            verbose         = GLOBAL_OPTS['verbose'],
            print_every     = GLOBAL_OPTS['print_every'],
            save_every      = GLOBAL_OPTS['save_every']
        )

        return t
    else:
        raise ValueError("Trainer type [%s] not implemented" % str(trainer_type))


# ======== Superconvergence example ======== #
# TODO : need to compare against a version that doesn't use this technique
def main() -> None:
    model = get_model(GLOBAL_OPTS['model'])
    t = get_trainer(model, "superconverge_cifar10_", trainer_type="cifar")
    lr_finder = get_lr_finder(
        t,
        1e-6,
        1.0,
        find_num_epochs = GLOBAL_OPTS['find_num_epochs'],
        max_batches=GLOBAL_OPTS['lr_max_batches']
    )
    find_start_time = time.time()
    lr_finder.find()
    lr_min, lr_max = lr_finder.get_lr_range()
    find_end_time = time.time()
    find_total_time = find_end_time - find_start_time
    print('Found learning rate range as %.4f -> %.4f' % (lr_min, lr_max))
    print('Learnig rate search took %s' % str(timedelta(seconds=find_total_time)))

    if GLOBAL_OPTS['sched_stepsize'] > 0:
        stepsize = GLOBAL_OPTS['sched_stepsize']
    else:
        stepsize = int(len(t.train_loader) / 2)

    # get a scheduler for learning rate
    lr_scheduler = get_scheduler(
        lr_min,
        lr_max,
        stepsize,
        sched_type='TriangularScheduler'
    )
    # get a scheduler for momentum
    mtm_scheduler = get_scheduler(
        #lr_min,
        #lr_max,
        # TODO : can we find these values by some procedure?
        0.80,
        0.95,
        stepsize,
        sched_type='InvTriangularScheduler'
    )
    t.set_lr_scheduler(lr_scheduler)
    t.set_mtm_scheduler(mtm_scheduler)
    train_start_time = time.time()
    t.train()
    train_end_time = time.time()
    train_total_time = train_end_time - train_start_time
    run_total_time = train_end_time - find_start_time
    print('Total training time (exclusive of find) : %s' %\
          str(timedelta(seconds=train_total_time))
    )
    print('Total training time (inclusive of find) : %s' %\
          str(timedelta(seconds=run_total_time))
    )
    max_acc, max_idx = t.get_max_acc()
    print('Maximum accuracy was %f at epoch %d' % (max_acc, max_idx))
    min_loss, min_idx = t.get_min_loss()
    print('Minimum loss was %f at batch %d' % (min_loss, min_idx))

    # plot outputs
    loss_title   = 'CIFAR10 Superconvergence Loss'
    acc_title    = 'CIFAR10 Superconvergence Accuracy'
    fig_filename = 'figures/cifar10_superconv_test.png'
    train_fig, train_ax = vis_loss_history.get_figure_subplots(2)
    vis_loss_history.plot_train_history_2subplots(
        train_ax,
        t.get_loss_history(),
        acc_history = t.get_acc_history(),
        cur_epoch = t.cur_epoch,
        iter_per_epoch = t.iter_per_epoch,
        loss_title = loss_title,
        acc_title = acc_title
    )
    train_fig.tight_layout()
    train_fig.savefig(fig_filename)


def get_parser() -> argparse.ArgumentParser:
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
                        default=20,
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
    parser.add_argument('--lr-max-batches',
                        type=int,
                        default=0,
                        help='Only run learning rate search for this many batches \
                        0 runs over all batches in dataset (default: 0)'
                        )
    parser.add_argument('--stop-when-acc',
                        type=float,
                        default=0.0,
                        help='Stop training when acc reaches this value (default: 0.0)'
                        )
    # Schedule options
    parser.add_argument('--exp-decay',
                        type=float,
                        default=0.001,
                        help='Exponential decay term'
                        )
    parser.add_argument('--sched-stepsize',
                        type=int,
                        default=0,
                        help='Size of step for learning rate scheduler'
                        )
    parser.add_argument('--sched-type',
                        type=str,
                        default='TriangularScheduler',
                        help='Scheduler to use (default: TriangularScheduler)'
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

    main()
