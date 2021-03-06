"""
EX_TRAIN_RESNETS
Train a resnet classifier on CIFAR10 (for now)

Stefan Wong 2019
"""

import os
import argparse
import time
from datetime import timedelta
# Tensorboard
from torch.utils import tensorboard
# lernomatic
from lernomatic.train import resnet_trainer
from lernomatic.models import resnets
from lernomatic.options import options
from lernomatic.util import expr_util


GLOBAL_OPTS = dict()


def main() -> None:
    # Get a model
    model = resnets.WideResnet(
        depth=GLOBAL_OPTS['num_layers'],
        num_classes=GLOBAL_OPTS['num_classes'],
        w_factor = GLOBAL_OPTS['widen_factor']
    )

    # Get a trainer
    # NOTE: no need for special dataset opts for now
    trainer = resnet_trainer.ResnetTrainer(
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
        if not os.path.isdir(GLOBAL_OPTS['tensorboard_dir']):
            os.mkdir(GLOBAL_OPTS['tensorboard_dir'])

        writer = tensorboard.SummaryWriter(log_dir=GLOBAL_OPTS['tensorboard_dir'])
        trainer.set_tb_writer(writer)

    # Optionally do a search pass here and add a scheduler
    if GLOBAL_OPTS['find_lr']:
        lr_finder = expr_util.get_lr_finder(trainer)
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
        stepsize = trainer.get_num_epochs() * (len(trainer.train_loader) // 2)
        # get scheduler
        lr_scheduler = expr_util.get_scheduler(
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
    parser.add_argument('--num-layers',
                        type=int,
                        default=28,
                        help='Number of layers for Resnet model'
                        )
    parser.add_argument('--num-classes',
                        type=int,
                        default=10,
                        help='Number of output classes for classification'
                        )
    parser.add_argument('--widen-factor',
                        type=int,
                        default=1,
                        help='Widen factor for wide Resnet'
                        )
    # Data options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='resnet-cifar10',
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
        for k, v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    main()
