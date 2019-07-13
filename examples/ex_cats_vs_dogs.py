"""
EX_CATS_VS_DOGS
Example of Kaggle Cats vs Dogs challenge

Stefan Wong 2019
"""

import os
import argparse
import torchvision
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
# models, etc
from lernomatic.data import hdf5_dataset
from lernomatic.models import cvdnet
from lernomatic.train import trainer
from lernomatic.train.cvd import cvd_trainer
# vis
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()

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

        cvd_test_dataset = hdf5_dataset.HDF5Dataset(
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

        cvd_test_dir = '/home/kreshnik/ml-data/cats-vs-dogs/test'
        cvd_test_dataset = datasets.ImageFolder(
            cvd_test_dir,
            test_dataset_transform
        )

    # get a network
    model = cvdnet.CVDNet2()

    #cvd_train = trainer.Trainer(
    cvd_train = cvd_trainer.CVDTrainer(
        model,
        # dataset options
        train_dataset   = cvd_train_dataset,
        test_dataset    = cvd_test_dataset,
        # training options
        loss_function   = 'CrossEntropyLoss',
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        momentum        = GLOBAL_OPTS['momentum'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        batch_size      = GLOBAL_OPTS['batch_size'],
        test_batch_size = GLOBAL_OPTS['test_batch_size'],
        # checkpoint
        checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
        save_every      = GLOBAL_OPTS['save_every'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # other
        print_every     = GLOBAL_OPTS['print_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    cvd_train.train()

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
    parser.add_argument('--momentum',
                        type=float,
                        default=0.0,
                        help='Momentum decay to use for optimizer'
                        )
    parser.add_argument('--learning-rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate for optimizer'
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
