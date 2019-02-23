"""
CELEBA_PROC_DATA
Process CelebA dataset into HDF5 file

TODO: Worry about this later, just load with ImageFolder for first experiment

Stefan Wong 2019
"""

import sys
import argparse
import h5py

# debug
from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()


def main():

    with h5py.File(GLOBAL_OPTS['outfile'], 'w') as fp:
        pass


def arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-root',
                        type=str,
                        default='/mnt/ml-data/datasets/celeba/img_align_celeba/',
                        help='Path to root of data'
                        )

    parser.add_argument('outfile',
                        type=str,
                        default='hdf5/celeba.h5',
                        help='Filename for output HDF5 file'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )

    return parser

if __name__ == '__main__':
    parser = arg_parser()
    args = vars(parser.parse_args())

    for k, v in args.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['input'] is None:
        print('ERROR: no input file specified.')
        sys.exit(1)

    main()
