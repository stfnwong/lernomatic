"""
EX_TRAIN_MNIST_AUTO
Train an Autoencoder on MNIST dataset

Stefan Wong 2019
"""

import argparse
import time
from datetime import timedelta
from lernomatic.train.autoencoder import mnist_auto_trainer
from lernomatic.models.autoencoder import mnist_autoencoder
from lernomatic.options import options


GLOBAL_OPTS = dict()


def main() -> None:
    # Get a model
    model = mnist_autoencoder.MNISTAutoencoder()

    # Get a trainer
    trainer = mnist_auto_trainer.MNISTAutoTrainer(
        model,
        # training parameters
        batch_size      = GLOBAL_OPTS['batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        momentum        = GLOBAL_OPTS['momentum'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
        save_img_dir    = GLOBAL_OPTS['save_img_dir'],
        # display,
        print_every     = GLOBAL_OPTS['print_every'],
        save_every      = GLOBAL_OPTS['save_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )
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
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--save-img-dir',
                        type=str,
                        default='./figures/',
                        help='Directory to save images to (default: figures/)'
                        )
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='mnist_autoencoder_',
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
