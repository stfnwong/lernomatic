"""
EX_CATS_VS_DOGS
Example of Kaggle Cats vs Dogs challenge

Stefan Wong 2019
"""

import os
import argparse
import torchvision
import time
from datetime import timedelta
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
# Tensorboard
from torch.utils import tensorboard
# models, etc
from lernomatic.data import hdf5_dataset
from lernomatic.train import trainer
from lernomatic.train.cvd import cvd_trainer
from lernomatic.models.cvd import cvdnet
from lernomatic.options import options
from lernomatic.util import expr_util
# vis
from lernomatic.vis import vis_loss_history


GLOBAL_OPTS = dict()
GLOBAL_USE_HDF5 = True      # until I figure out what is up with HDF5 data


def main() -> None:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    train_dataset_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset_transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # HDF5 Datasets
    if GLOBAL_USE_HDF5 is True:
        cvd_train_dataset = hdf5_dataset.HDF5Dataset(
            GLOBAL_OPTS['train_dataset'],
            feature_name = 'images',
            label_name = 'labels',
            label_max_dim = 1,
            transform=normalize
        )

        cvd_val_dataset = hdf5_dataset.HDF5Dataset(
            GLOBAL_OPTS['test_dataset'],
            feature_name = 'images',
            label_name = 'labels',
            label_max_dim = 1,
            transform=normalize
        )
    else:
        cvd_train_dir = '/home/kreshnik/ml-data/cats-vs-dogs/train'
        # ImageFolder dataset
        cvd_train_dataset = datasets.ImageFolder(
            cvd_train_dir,
            train_dataset_transform
        )

        csv_val_dir = '/home/kreshnik/ml-data/cats-vs-dogs/test'
        cvd_val_dataset = datasets.ImageFolder(
            csv_val_dir,
            test_dataset_transform
        )

    # get a network
    model = cvdnet.CVDNet2()
    cvd_train = cvd_trainer.CVDTrainer(
        model,
        # dataset options
        train_dataset   = cvd_train_dataset,
        val_dataset     = cvd_val_dataset,
        # training options
        loss_function   = 'CrossEntropyLoss',
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        momentum        = GLOBAL_OPTS['momentum'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        batch_size      = GLOBAL_OPTS['batch_size'],
        val_batch_size  = GLOBAL_OPTS['val_batch_size'],
        # checkpoint
        checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
        save_every      = GLOBAL_OPTS['save_every'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # other
        print_every     = GLOBAL_OPTS['print_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    # Add a tensorboard writer
    if GLOBAL_OPTS['tensorboard_dir'] is not None:
        if not os.path.isdir(GLOBAL_OPTS['tensorboard_dir']):
            os.mkdir(GLOBAL_OPTS['tensorboard_dir'])
        writer = tensorboard.SummaryWriter(log_dir=GLOBAL_OPTS['tensorboard_dir'])
        cvd_train.set_tb_writer(writer)

    # Optionally do a search pass here and add a scheduler
    if GLOBAL_OPTS['find_lr']:
        lr_finder = expr_util.get_lr_finder(cvd_train)
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
        stepsize = cvd_train.get_num_epochs() * len(trainer.train_loader) // 2
        # get scheduler
        lr_scheduler = expr_util.get_scheduler(
            lr_find_min,
            lr_find_max,
            stepsize,
            sched_type='TriangularScheduler'
        )
        cvd_train.set_lr_scheduler(lr_scheduler)

    # train the model
    train_start_time = time.time()
    cvd_train.train()
    train_end_time = time.time()
    train_total_time = train_end_time - train_start_time

    print('Total scheduled training time [%s] (%d epochs)  %s' %\
            (repr(cvd_train), cvd_train.cur_epoch,
             str(timedelta(seconds = train_total_time)))
    )

    # Show results
    fig, ax = vis_loss_history.get_figure_subplots(num_subplots=2)
    vis_loss_history.plot_train_history_2subplots(
        ax,
        cvd_train.get_loss_history(),
        acc_history = cvd_train.get_acc_history(),
        iter_per_epoch = cvd_train.iter_per_epoch,
        loss_title = 'CVD Loss',
        acc_title = 'CVD Acc',
        cur_epoch = cvd_train.cur_epoch
    )

    fig.savefig('figures/cvd_train.png')


def get_parser() -> argparse.ArgumentParser:
    parser = options.get_trainer_options()
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
    # Dataset options
    parser.add_argument('--train-dataset',
                        type=str,
                        default='hdf5/cvd_train.h5',
                        help='Path to training dataset'
                        )
    parser.add_argument('--test-dataset',
                        type=str,
                        default='hdf5/cvd_test.h5',
                        help='Path to test dataset'
                        )
    parser.add_argument('--val-dataset',
                        type=str,
                        default='hdf5/cvd_val.h5',
                        help='Path to validation dataset'
                        )
    # Data options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='cvd_',
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
