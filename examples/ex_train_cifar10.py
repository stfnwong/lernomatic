"""
EX_TRAIN_CIFAR10
Train a classifier on the CIFAR10 dataset

Stefan Wong 2019
"""

import sys
import argparse
# Tensorboard
import torchvision
from torch.utils import tensorboard
# lernomatic
from lernomatic.train import cifar_trainer
from lernomatic.models import cifar
from lernomatic.vis import vis_loss_history
from lernomatic.options import options
from lernomatic.util import expr_util

import time
from datetime import timedelta

GLOBAL_OPTS = dict()


def main() -> None:
    # Get a model
    model = cifar.CIFAR10Net()
    # Get a trainer
    trainer = cifar_trainer.CIFAR10Trainer(
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
        stepsize = trainer.get_num_epochs() * len(trainer.train_loader) // 2
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

    # Visualise the output
    train_fig, train_ax = vis_loss_history.get_figure_subplots()
    vis_loss_history.plot_train_history_2subplots(
        train_ax,
        trainer.get_loss_history(),
        acc_history = trainer.get_acc_history(),
        cur_epoch = trainer.cur_epoch,
        iter_per_epoch = trainer.iter_per_epoch,
        loss_title = 'CIFAR-10 Training Loss',
        acc_title = 'CIFAR-10 Training Accuracy '
    )
    train_fig.savefig(GLOBAL_OPTS['fig_name'], bbox_inches='tight')

    # TODO : infer on some test data?


def get_parser() -> argparse.ArgumentParser:
    parser = options.get_trainer_options()
    # add some extra options for this particular example
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
    # Figure output
    parser.add_argument('--fig-name',
                        type=str,
                        default='figures/cifar10net_train.png',
                        help='Name of file to place output figure into'
                        )
    # Checkpoint options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='cifar10',
                        help='Name to prepend to all checkpoints'
                        )
    # TODO : implement this...
    parser.add_argument('--load-checkpoint',
                        type=str,
                        default=None,
                        help='Load a given checkpoint'
                        )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS (%s) ---- ' % str(sys.argv[0]))
        for k,v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    main()
