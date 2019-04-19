"""
CVD_CATS_VS_DOGS
Process data for Cats vs Dogs
"""

import os
import argparse
import h5py
import numpy as np

from lernomatic.data import data_split
from lernomatic.data.cvd import image_proc

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def main():

    # discover files in each path
    if GLOBAL_OPTS['data_root'] is not None and GLOBAL_OPTS['data_root'] != '':
        cats_path = GLOBAL_OPTS['data_root'] + GLOBAL_OPTS['cats_path']
        dogs_path = GLOBAL_OPTS['data_root'] + GLOBAL_OPTS['dogs_path']
    else:
        cats_path = GLOBAL_OPTS['cats_path']
        dogs_path = GLOBAL_OPTS['dogs_path']

    cat_list = [cats_path + path for path in os.listdir(cats_path)]
    dog_list = [dogs_path + path for path in os.listdir(dogs_path)]
    cat_ids  = os.listdir(cats_path)
    dog_ids  = os.listdir(dogs_path)

    label_list = [GLOBAL_OPTS['cat_label']] * len(cat_list) +\
                 [GLOBAL_OPTS['dog_label']] * len(dog_list)
    #label_list = [np.asarray([1, 0])] * len(cat_list) +\
    #             [np.asarray([0, 1])] * len(dog_list)

    id_list = []
    for data in cat_ids + dog_ids:
        id_list.append(os.path.splitext(data)[0])

    # Prune out images that we can't load
    if GLOBAL_OPTS['verbose']:
        print('Checking images....')

    path_list = cat_list + dog_list
    import cv2
    num_pruned = 0
    for n, path in enumerate(path_list):
        # display progress
        if GLOBAL_OPTS['verbose']:
            print('Checking image [%d / %d] ' % (n+1, len(path_list)), end=' ')

        img = cv2.imread(path)
        if img is None:
            if GLOBAL_OPTS['verbose']:
                print('  Failed to load image [%s] ' % str(path), end=' ')
            path_list.pop(n)
            label_list.pop(n)
            id_list.pop(n)
            num_pruned += 1

        if GLOBAL_OPTS['verbose']:
            print(' ', end='\r')

    if GLOBAL_OPTS['verbose']:
        print('\n done (pruned %d images from total, %d remaining)' % (num_pruned, len(path_list)))

    # Split the data
    splitter = data_split.ListSplitter(
        split_ratios = GLOBAL_OPTS['split_ratios'],
        split_names = GLOBAL_OPTS['split_names'],
        data_root = GLOBAL_OPTS['data_root'],
        verbose = GLOBAL_OPTS['verbose']
    )

    splits = splitter.gen_splits(
        path_list,
        label_list,
        id_list = id_list
    )

    if GLOBAL_OPTS['verbose']:
        for s in splits:
            print(s)

    # Process each split
    proc = image_proc.ImageDataProc(
        label_dataset_size = 1,
        verbose = GLOBAL_OPTS['verbose']
    )

    outfiles = [GLOBAL_OPTS['train_data'], GLOBAL_OPTS['test_data'], GLOBAL_OPTS['val_data']]
    for n, s in enumerate(splits):
        print('Processing split <%s>' % repr(s))
        proc.proc(s, outfiles[n])



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
                        default='Cat/',
                        help='Path to cat images'
                        )
    parser.add_argument('--dogs-path',
                        type=str,
                        default='Dog/',
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
    parser.add_argument('--data-root',
                        type=str,
                        default='/mnt/ml-data/datasets/cats-vs-dogs/PetImages/',
                        help='Data path root'
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
    # split options
    parser.add_argument('--split-names',
                        type=str,
                        default='train,test,val',
                        help='Comma seperated list of names for each split (default: train,test,val)'
                        )
    parser.add_argument('--split-ratios',
                        type=str,
                        default='0.7,0.15,0.15',
                        help='Comma seperated list of ratios for each split. Must sum to 1 (default: 0.7,0.15,0.15)'
                        )
    parser.add_argument('--split-method',
                        type=str,
                        default='random',
                        help='Method to use when selecting data elements for split'
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

    GLOBAL_OPTS['split_names'] = GLOBAL_OPTS['split_names'].split(',')
    split_ratios = GLOBAL_OPTS['split_ratios'].split(',')
    split_ratio_floats = []

    for s in split_ratios:
        split_ratio_floats.append(float(s))

    GLOBAL_OPTS['split_ratios'] = split_ratio_floats

    if len(GLOBAL_OPTS['split_names']) != len(GLOBAL_OPTS['split_ratios']):
        raise ValueError('Number of split rations must equal number of split names')

    if sum(split_ratio_floats) > 1.0:
        raise ValueError('Sum of split ratios cannot exceed 1.0')

    main()
