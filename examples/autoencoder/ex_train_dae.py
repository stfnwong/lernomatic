"""
EX_TRAIN_DAE
Train a de-noising autoencoder

Stefan Wong 2019
"""


import torch
import torchvision
import argparse
import numpy as np
import matplotlib.pyplot as plt
# timing
import time
from datetime import timedelta

from lernomatic.train.autoencoder import dae_trainer
from lernomatic.models.autoencoder import denoise_ae
from lernomatic.models import common        # mainly for type hints


GLOBAL_OPTS = dict()



def main() -> None:
    # get some models
    encoder = denoise_ae.DAEEncoder()
    decoder = denoise_ae.DAEDecoder()

    trainer = dae_trainer.DAETrainer(
        encoder,
        decoder,
        # datasets
        train_dataset = train_dataset,
        val_dataset   = val_dataset,
        device_id     = GLOBAL_OPTS['device_id'],
        # trainer params
        batch_size = GLOBAL_OPTS['batch_size'],
        num_epochs = self.test_num_epochs,
        # disable saving
        save_every = 0,
        print_every = GLOBAL_OPTS['print_every'],
        verbose = self.verbose
    )
    # train the model
    train_start_time = time.time()
    trainer.train()
    train_end_time = time.time()
    train_total_time = train_end_time - train_start_time




def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--mode',
                        type=str,
                        default='unsupervised',
                        help='Tool mode. Must be one of %s (default: unsupervised)' % str(VALID_MODES)
                        )
    parser.add_argument('--data-dir',
                        type=str,
                        default='./data',
                        help='Path to location where data will be downloaded (default: ./data)'
                        )
    parser.add_argument('--print-every',
                        type=int,
                        default=10,
                        help='Print output every N epochs'
                        )
    # TODO: should -2 be every 2 epochs, -3 every 3 epochs, etc?
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
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )
    # Training options
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--val-batch-size',
                        type=int,
                        default=0,
                        help='Batch size to use during testing'
                        )
    parser.add_argument('--start-epoch',
                        type=int,
                        default=0,
                        )
    parser.add_argument('--num-epochs',
                        type=int,
                        default=40,
                        help='Epoch to stop training at'
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
                        default='aae',
                        help='Name to prepend to all checkpoints'
                        )

    return parser



if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    GLOBAL_OPTS['checkpoint_name'] =\
        str(GLOBAL_OPTS['checkpoint_name']) + '_' + GLOBAL_OPTS['mode']

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS ---- ')
        for k,v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    if GLOBAL_OPTS['mode'] == 'unsupervised':
        unsupervised()
    elif GLOBAL_OPTS['mode'] == 'semisupervised':
        semisupervised()
    elif GLOBAL_OPTS['mode'] == 'supervised':
        supervised()
    else:
        raise ValueError('Unsupported tool mode [%s]' % str(GLOBAL_OPTS['mode']))
