"""
COCO_HDF5_INSPECTOR
Look inside a COCO HDF5 file

Stefan Wong 2019
"""

import sys
import argparse

from lernomatic.util import hdf5_util


GLOBAL_OPTS = dict()

def main():

    hdf5_data = hdf5_util.HDF5Data(
        GLOBAL_OPTS['input'],
        verbose=GLOBAL_OPTS['verbose']
    )




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
    parser.add_argument('--feature-name',
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

    main()
