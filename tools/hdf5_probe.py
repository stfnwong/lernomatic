"""
HDF5_PROBE
Examine the contents of an HDF5 file

Stefan Wong 2018
"""

import sys
import argparse
from lernomatic.util import hdf5_util
from lernomatic.vis import vis_data

# debug
#

GLOBAL_OPTS = dict()

def inspect():
    # get hdf5 wrapper
    hdf5_data = hdf5_util.HDF5Data(
        GLOBAL_OPTS['input'],
        verbose=GLOBAL_OPTS['verbose']
    )
    dataset_meta = hdf5_data.dump_meta()
    print(hdf5_data)


def load():
    hdf5_data = hdf5_util.HDF5Data(
        GLOBAL_OPTS['input'],
        verbose=GLOBAL_OPTS['verbose']
    )
    hdf5_data.read(GLOBAL_OPTS['input'])


def show_pn():
    hdf5_data = hdf5_util.HDF5Data(
        GLOBAL_OPTS['input'],
        verbose=GLOBAL_OPTS['verbose']
    )

    labels = hdf5_data.get_dataset('labels')
    num_pos = 0
    num_neg = 0
    num_unk = 0
    for n, label in enumerate(labels):
        print('Scanning label [%d/%d] ' % (n+1, len(labels)), end='\r')
        if label == 1:
            num_pos += 1
        elif label == 0:
            num_neg += 1
        else:
            num_unk += 1

    print('\n\t [DONE]')
    print('Total labels            : %d' % len(labels))
    print('Positive examples       : %d (%.4f %%)' % (num_pos, 100.0 * (num_pos / len(labels))))
    print('Negative examples       : %d (%.4f %%)' % (num_neg, 100.0 * (num_neg / len(labels))))
    print('Uncategorised examples  : %d ' % num_unk)


def find():
    pass


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        help='Input file'
                        )
    parser.add_argument('--mode',
                        choices=['inspect',  'load', 'find'],
                        default='inspect',
                        help='Select the tool mode from one of inspect, load, find'
                        )
    parser.add_argument('--vis-index',
                        type=int,
                        default=0,
                        help='Index of dataset to visualize'
                        )
    parser.add_argument('--vis-dataset',
                        type=str,
                        default='feature',
                        help='Name of dataset containing data to visualize'
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

    # Figure out what to do (ie: what mode are we in?)
    if GLOBAL_OPTS['mode'] == 'inspect':
        inspect()
    elif GLOBAL_OPTS['mode'] == 'pn':
        show_pn()
    elif GLOBAL_OPTS['mode'] == 'find':
        find()

