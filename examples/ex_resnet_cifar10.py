"""
EX_RESNET_CIFAR10
Resnet trained on CIFAR-10 with LRFinder and Scheduling

Stefan Wong 2019
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
# Tensorboard
from torch.utils import tensorboard
# lernomatic
from lernomatic.models import common
from lernomatic.models import resnets
from lernomatic.train import resnet_trainer
from lernomatic.train import schedule
from lernomatic.param import lr_common
# some vis stuff
from lernomatic.vis import vis_loss_history
from lernomatic.options import options

import time
from datetime import timedelta

GLOBAL_OPTS = dict()


def get_model(depth:int=58, num_classes:int=10) -> common.LernomaticModel:
    model = resnets.WideResnet(
        depth = depth,
        num_classes = num_classes
    )

    return model


def get_trainer(model:common.LernomaticModel, checkpoint_name:str) -> resnet_trainer.ResnetTrainer:
    trainer = resnet_trainer.ResnetTrainer(
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


def main() -> None:
    # get a model and train it as a reference
    ref_model = get_model()
    ref_trainer = get_trainer(ref_model, 'ex_cifar10_lr_find_schedule_')

    if GLOBAL_OPTS['tensorboard_dir'] is not None:
        ref_writer = tensorboard.SummaryWriter(log_dir=GLOBAL_OPTS['tensorboard_dir'])
        ref_trainer.set_tb_writer(ref_writer)

    ref_train_start_time = time.time()
    ref_trainer.train()
    ref_train_end_time = time.time()
    ref_train_total_time = ref_train_end_time - ref_train_start_time
    print('Total reference training time [%s] (%d epochs)  %s' %\
            (repr(ref_trainer), ref_trainer.cur_epoch,
             str(timedelta(seconds = ref_train_total_time)))
    )

    # get a model and train it with a scheduler
    sched_model = get_model()
    sched_trainer = get_trainer(sched_model, 'ex_cifar10_lr_find_schedule_')

    if GLOBAL_OPTS['tensorboard_dir'] is not None:
        if not os.path.isdir(GLOBAL_OPTS['tensorboard_dir']):
            os.mkdir(GLOBAL_OPTS['tensorboard_dir'])
        sched_writer = tensorboard.SummaryWriter(log_dir=GLOBAL_OPTS['tensorboard_dir'])
        sched_trainer.set_tb_writer(sched_writer)

    # get an LRFinder object
    lr_finder = lr_common.LogFinder(
        sched_trainer,
        lr_min         = GLOBAL_OPTS['find_lr_min'],
        lr_max         = GLOBAL_OPTS['find_lr_max'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = GLOBAL_OPTS['find_print_every']
    )
    print(lr_finder)

    lr_find_start_time = time.time()
    lr_finder.find()
    lr_find_min, lr_find_max = lr_finder.get_lr_range()
    lr_find_end_time = time.time()
    lr_find_total_time = lr_find_end_time - lr_find_start_time
    print('Total parameter search time : %s' % str(timedelta(seconds = lr_find_total_time)))

    if GLOBAL_OPTS['verbose']:
        print('Found learning rate range as %.4f -> %.4f' % (lr_find_min, lr_find_max))

    # get a scheduler
    lr_sched_obj = getattr(schedule, GLOBAL_OPTS['sched_type'])
    lr_scheduler = lr_sched_obj(
        stepsize = int(len(sched_trainer.train_loader) / 4),
        lr_min = lr_find_min,
        lr_max = lr_find_max
    )
    assert(sched_trainer.acc_iter == 0)
    sched_trainer.set_lr_scheduler(lr_scheduler)

    sched_train_start_time = time.time()
    sched_trainer.train()
    sched_train_end_time = time.time()
    sched_train_total_time = sched_train_end_time - sched_train_start_time
    print('Total scheduled training time [%s] (%d epochs)  %s' %\
            (repr(sched_trainer), ref_trainer.cur_epoch,
             str(timedelta(seconds = sched_train_total_time)))
    )
    print('Scheduled training time (including find time) : %s' %\
          str(timedelta(seconds = sched_train_total_time + lr_find_total_time))
    )

    # Compare loss, accuracy
    fig, ax = vis_loss_history.get_figure_subplots(2)
    # TODO : write out figure...

    # Could also have a step here to show inference



def get_parser() -> argparse.ArgumentParser:
    parser = options.get_trainer_options()
    parser = options.get_lr_finder_options(parser)
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
    # Data options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--load-checkpoint',
                        type=str,
                        default=None,
                        help='Load a given checkpoint'
                        )
    parser.add_argument('--tensorboard-dir',
                        default=None,
                        type=str,
                        help='Directory to save tensorboard runs to. If None, tensorboard is not used. (default: None)'
                        )
    parser.add_argument('--sched-type',
                        type=str,
                        default='TriangularScheduler',
                        help='Type of learning rate scheduler to use. Must be the name of a class in lernomatic.train.schedule. (default: TriangularScheduler)'
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
