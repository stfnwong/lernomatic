"""
CVD_CATS_VS_DOGS
Process data for Cats vs Dogs
"""

import os
import argparse
import h5py
import cv2

from lernomatic.data import data_split

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

def main():

    # discover files in each path
    cat_list = os.listdir(GLOBAL_OPTS['cats_path'])
    dog_list = os.listdir(GLOBAL_OPTS['dogs_path'])

    label_list = [GLOBAL_OPTS['cat_label']] * len(cat_list) +\
                 [GLOBAL_OPTS['dog_label']] * len(dog_list)


   # TODO : need a splitter object here to generate datasets

    print('%d dogs, %d cats, %d elements, %d labels' % \
          (len(dog_list), len(cat_list),
           len(dog_list) + len(cat_list),
           len(label_list))
    )


def get_parser():
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
    # input data paths
    parser.add_argument('--cats-path',
                        type=str,
                        default=None,
                        help='Path to cat images'
                        )
    parser.add_argument('--dogs-path',
                        type=str,
                        default=None,
                        help='Path to dog images'
                        )
    parser.add_argument('--cat-label',
                        type=int,
                        default=0,
                        help='Label (integer) to use for cats'
                        )
    parser.add_argument('--dog-label',
                        type=int,
                        default=1,
                        help='Label (integer) to use for dogs'
                        )
    # output options
    parser.add_argument('--train-data',
                        type=str,
                        default='hdf5/cvd_train.h5',
                        help='Training data output file'
                        )
    parser.add_argument('--test-data',
                        type=str,
                        default='hdf5/cvd_test.h5',
                        help='Test data output file'
                        )
    parser.add_argument('--val-data',
                        type=str,
                        default='hdf5/cvd_val.h5',
                        help='Validation data output file'
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
            print('%s : %s' % (str(k), str(v)))
    main()
