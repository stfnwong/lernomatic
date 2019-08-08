"""
OPTIONS
Various standard options that can be added to examples, tests, etc

Stefan Wong 2019
"""
# TODO: lr_finder options....

import argparse


def get_basic_options(parser:argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )

    return parser


def get_trainer_options(parser:argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    # Display and save options
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
    parser.add_argument('--save-best',
                        action='store_true',
                        default=False,
                        help='Save the best weights in addition to any checkpoints'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of workers to use when generating HDF5 files'
                        )
    # optimization
    parser.add_argument('--loss-function',
                        type=str,
                        default='CrossEntropyLoss',
                        help='Loss function to use. Must be the name of a class in torch.nn (default: CrossEntropyLoss)'
                        )
    parser.add_argument('--optim-function',
                        type=str,
                        default='Adam',
                        help='Optim function to use. Must be the name of a class in torch.optim (default: Adam)'
                        )
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )
    # Training options
    parser.add_argument('--start-epoch',
                        type=int,
                        default=0,
                        help='Epoch to start training from'
                        )
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Epoch to stop training at'
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--val-batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during validation'
                        )
    parser.add_argument('--weight-decay',
                        type=float,
                        default=1e-4,
                        help='Weight decay to use for optimizer'
                        )
    parser.add_argument('--learning-rate',
                        type=float,
                        default=3e-4,
                        help='Learning rate for ADAM optimizer'
                        )
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        help='Momentum for ADAM'
                        )
    parser.add_argument('--grad-clip',
                        type=float,
                        default=5.0,
                        help='Clip gradients at this (absolute) value'
                        )

    return parser
