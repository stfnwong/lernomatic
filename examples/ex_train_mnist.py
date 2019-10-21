"""
EX_TRAIN_MNIST
Train a classifier on the MNIST handwritten digits example

Stefan Wong 2019
"""

import argparse
import time
from datetime import timedelta
# Tensorboard
import torchvision
from torch.utils import tensorboard
# lernomatic
from lernomatic.train import mnist_trainer
from lernomatic.train import schedule
from lernomatic.models import mnist
from lernomatic.options import options
# params
from lernomatic.param import lr_common


GLOBAL_OPTS = dict()

# Helper function for finder
def get_lr_finder(trainer, find_type='LogFinder') -> lr_common.LRFinder:
    if not hasattr(lr_common, find_type):
        raise ValueError('Unknown learning rate finder type [%s]' % str(find_type))

    lr_find_obj = getattr(lr_common, find_type)
    lr_finder = lr_find_obj(
        trainer,
        lr_min         = GLOBAL_OPTS['find_lr_min'],
        lr_max         = GLOBAL_OPTS['find_lr_max'],
        lr_select_method = GLOBAL_OPTS['find_lr_select_method'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = GLOBAL_OPTS['find_print_every']
    )

    return lr_finder


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


# ======== EXPERIMENT ======== #
def main() -> None:
    # Get a model
    model = mnist.MNISTNet()
    # Get a trainer
    trainer = mnist_trainer.MNISTTrainer(
        model,
        # training parameters
        batch_size = GLOBAL_OPTS['batch_size'],
        num_epochs = GLOBAL_OPTS['num_epochs'],
        learning_rate = GLOBAL_OPTS['learning_rate'],
        momentum = GLOBAL_OPTS['momentum'],
        weight_decay = GLOBAL_OPTS['weight_decay'],
        # device
        device_id = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
        # display,
        print_every = GLOBAL_OPTS['print_every'],
        save_every = GLOBAL_OPTS['save_every'],
        verbose = GLOBAL_OPTS['verbose']
    )

    if GLOBAL_OPTS['tensorboard_dir'] is not None:
        writer = tensorboard.SummaryWriter()
        trainer.set_tb_writer(writer)

    # Optionally do a search pass here and add a scheduler
    if GLOBAL_OPTS['find_lr']:
        lr_finder = get_lr_finder(trainer)
        lr_find_start_time = time.time()
        lr_finder.find()
        lr_find_min, lr_find_max = lr_finder.get_lr_range()
        lr_find_end_time = time.time()
        lr_find_total_time = lr_find_end_time - lr_find_start_time
        print('Found learning rate range %.4f -> %.4f' % (lr_find_min, lr_find_max))
        print('Total find time [%s] ' %\
                str(timedelta(seconds = lr_find_total_time))
        )

        # Now get a scheduler
        stepsize = len(trainer.train_loader) // 2
        # get scheduler
        lr_scheduler = get_scheduler(
            lr_find_min,
            lr_find_max,
            stepsize,
            sched_type='TriangularScheduler'
        )

        trainer.set_lr_scheduler(lr_scheduler)

    # train the model
    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    train_total_time = train_end_time - train_start_time

    print('Total training time [%s] (%d epochs)  %s' %\
            (repr(trainer), trainer.cur_epoch,
             str(timedelta(seconds = train_total_time)))
    )


def get_parser() -> argparse.ArgumentParser:
    parser = options.get_trainer_options()
    parser = options.get_lr_finder_options(parser)
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--find-lr',
                        action='store_true',
                        default=False,
                        help='Search for optimal learning rate'
                        )
    # Data options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='mnist',
                        help='Name to prepend to all checkpoints'
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
