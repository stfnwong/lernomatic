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

# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()
VALID_SPLITS = ('all', 'train', 'test', 'val')


def proc_split(split_path:str, split_outfile_name:str, split_name:str, label_file:str=None) -> None:

    split_data = data_split.DataSplit(split_name=split_name)
    # get labels from file
    if label_file is not None:
        label_list = []
        with open(label_file, 'r') as fp:
            for line in fp:
                label_list.append(line.strip())

        file_list = os.listdir(GLOBAL_OPTS['dataset_root'] + split_path)
        print('Found %d files in folder [%s]' % (len(file_list), str(GLOBAL_OPTS['dataset_root'] + split_path)))

        if len(label_list) != len(file_list):
            raise ValueError('Number of labels (%d) does not match number of files (%d)' %\
                             (len(label_list), len(file_list))
            )

        # Now add the data to the split
        for n, (label, fname) in enumerate(zip(label_list, file_list)):
            print('Adding element [%d / %d] <%d> to split <%s>' %\
                  (n, len(label_list), int(label), split_data.split_name), end='\r'
            )
            path = GLOBAL_OPTS['dataset_root'] + split_path + str(fname)
            label = int(label)
            item_id = np.string_(str(label))
            split_data.add_path(path)
            split_data.add_label(label)
            split_data.add_id(item_id)

    # get labels from folder names
    else:
        imagenet_label_folders = os.listdir(GLOBAL_OPTS['dataset_root'] + split_path)
        if len(imagenet_label_folders) != 1000:
            raise ValueError('Expected 1000 class folders in imagenet, got %d folders' %\
                            len(imagenet_label_folders)
            )

        print('Found %d folders in path [%s]' % (len(imagenet_label_folders), str(GLOBAL_OPTS['dataset_root'])))

        folder_map = dict()
        total = 0
        for n, folder in enumerate(imagenet_label_folders):
            folder_map[folder] = os.listdir(GLOBAL_OPTS['dataset_root'] + split_path + folder)
            print('Processing folder [%d / %d] (%d items) ' %\
                (n, len(imagenet_label_folders), len(folder_map[folder])), end='\r'
            )
            total += len(folder_map[folder])

        print('\nFound %d items total' % total)

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

        # cycle over the keys, iterate over each label until we run out of data
        num_elem = 0
        while num_elem < total:
            for k in folder_map.keys():
                if label_ptr[k] < label_dims[k]:
                    path = GLOBAL_OPTS['dataset_root'] + split_path + str(k) + '/' +  str(folder_map[k][idx_map[k][label_ptr[k]]])
                    label = str(k[1:] + '-' + str(k[0]))
                    label_ptr[k] += 1
                    split_data.add_path(path)
                    split_data.add_label(label_map[k])
                    split_data.add_id(np.string_(label))
                    num_elem += 1
                    print('Added element [%d / %d] with label [%s] (%d of %d total)' %\
                        (label_ptr[k], label_dims[k], str(k), num_elem, total),
                        end='\r'
                    )

    print('\n Split <%s> now contains %d elements' % (split_data.split_name, len(split_data)))

    # Get an image processor to process images
    processor = image_proc.ImageDataProc(
        image_dataset_name = GLOBAL_OPTS['image_dataset_name'],
        image_dataset_size = (3, 256, 256),
        label_dataset_name = GLOBAL_OPTS['label_dataset_name'],
        label_dataset_size = len(split_data.elem_ids[0]),
        id_dtype = 'S10',
        verbose = GLOBAL_OPTS['verbose']
    )
    processor.proc(split_data, split_outfile_name)


def proc() -> None:

    if GLOBAL_OPTS['split_name'] == 'all':
        splits = ['train', 'test', 'val']
    else:
        splits = [str(GLOBAL_OPTS['split_name'])]

    for n, split in enumerate(splits):
        path = str(split) + '/'
        h5_name = 'ILSVRC_' + str(split).upper() + '.h5'
        split_name = 'imagenet_' + str(split)
        print('Processing split [%d / %d] (%s)' % (n, len(splits), split_name))
        if split == 'val':
            proc_split(path, h5_name, split_name, GLOBAL_OPTS['val_label_path'])
        else:
            proc_split(path, h5_name, split_name)


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--split-name',
                        type=str,
                        default='all',
                        help='Name of split to generate. Valid options are %s (default: all)' % str(VALID_SPLITS)
                        )
    parser.add_argument('--val-label-path',
                        type=str,
                        default=None,
                        help='Path to validation data'
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

    if GLOBAL_OPTS['split_name'] not in VALID_SPLITS:
        raise ValueError('Invalid split name [%s], must be one of %s' %\
                         (str(GLOBAL_OPTS['split_name']), str(VALID_SPLITS))
        )

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS ---- ')
        for k,v in GLOBAL_OPTS.items():
            print('%s : %s' % (str(k), str(v)))


    proc()
