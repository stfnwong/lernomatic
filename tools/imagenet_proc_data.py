"""
IMAGENET_PROC_DATA
Process the imagenet dataset into an HDF5 file

Stefan Wong 2019
"""

import sys
import os
import argparse
import h5py
import numpy as np
from collections import OrderedDict

from lernomatic.data import image_proc
from lernomatic.data import data_split


#debufg
from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()


def main() -> None:
    imagenet_label_folders = os.listdir(GLOBAL_OPTS['dataset_root'] + 'train/')
    if len(imagenet_label_folders) != 1000:
        raise ValueError('Expected 1000 class folders in imagenet, got %d folders' %\
                         len(imagenet_label_folders)
        )

    print('Found %d folders in path [%s]' % (len(imagenet_label_folders), str(GLOBAL_OPTS['dataset_root'])))

    folder_map = dict()
    total = 0
    for n, folder in enumerate(imagenet_label_folders):
        folder_map[folder] = os.listdir(GLOBAL_OPTS['dataset_root'] + 'train/' + folder)
        print('Processing folder [%d / %d] (%d items) ' %\
              (n, len(imagenet_label_folders), len(folder_map[folder])), end='\r'
        )
        total += len(folder_map[folder])

    print('\nFound %d items total' % total)

    # TODO : this is just a test for re-writing the labels later
    #for k in folder_map.keys():
    #    print(str(k[1:] + '-' + str(k[0])))

    # Randomly shuffle data into datasplit object
    idx_map    = dict()         # randomly permute the indexs
    label_dims = dict()         # number of items for label k
    label_ptr  = dict()         # points to current index of label k
    for k, v in folder_map.items():
        idx_map[k] = np.random.permutation(range(len(v)))
        label_dims[k] = len(v)
        label_ptr[k] = 0

    # generate label map
    label_map = OrderedDict()
    n = 0
    for k in folder_map.keys():
        label_map[k] = n
        n += 1

    # TODO : genrealize
    train_split = data_split.DataSplit(split_name='imagenet_train')
    # cycle over the keys, iterate over each label until we run out of data
    num_elem = 0
    while num_elem < total:
        for k in folder_map.keys():
            if label_ptr[k] < label_dims[k]:
                path = GLOBAL_OPTS['dataset_root'] + 'train/' + str(k) + '/' +  str(folder_map[k][idx_map[k][label_ptr[k]]])
                label = str(k[1:] + '-' + str(k[0]))
                label_ptr[k] += 1
                train_split.add_path(path)
                train_split.add_label(label_map[k])
                train_split.add_id(np.string_(label))
                num_elem += 1
                print('Added element [%d / %d] with label [%s] (%d of %d total)' %\
                      (label_ptr[k], label_dims[k], str(k), num_elem, total),
                      end='\r'
                )

    print('\n Split <%s> now contains %d elements' % (train_split.split_name, len(train_split)))

    # Get an image processor to process images
    processor = image_proc.ImageDataProc(
        image_dataset_name = GLOBAL_OPTS['image_dataset_name'],
        image_dataset_size = (3, 256, 256),
        label_dataset_name = GLOBAL_OPTS['label_dataset_name'],
        label_dataset_size = len(train_split.elem_ids[0]),
        id_dtype = 'S10',
        verbose = GLOBAL_OPTS['verbose']
    )
    processor.proc(train_split, 'ILSVRC_TRAIN.h5')




def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    #parser.add_argument('input',
    #                    type=str,
    #                    help='Input file'
    #                    )
    parser.add_argument('--mode',
                        choices=['inspect',  'load', 'find'],
                        default='inspect',
                        help='Select the tool mode from one of inspect, load, find'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--dataset-root',
                        type=str,
                        default='/mnt/ml-data/datasets/ILSVRC/Data/CLS-LOC/',
                        help='Path to root directory containing imagenet data'
                        )

    # Dataset options
    parser.add_argument('--image-dataset-name',
                        type=str,
                        default='images',
                        help='Name to use for image dataset'
                        )
    parser.add_argument('--label-dataset-name',
                        type=str,
                        default='labels',
                        help='Name to use for label dataset'
                        )

    return parser

if __name__ == '__main__':
    parser = arg_parser()
    args = vars(parser.parse_args())

    for k, v in args.items():
        GLOBAL_OPTS[k] = v

    #if GLOBAL_OPTS['input'] is None:
    #    print('ERROR: no input file specified.')
    #    sys.exit(1)

    main()
