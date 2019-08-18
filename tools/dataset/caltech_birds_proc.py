"""
CALTECH_BIRDS_PROC
Process the Caltech Birds-200-2011 dataset for use with DCGAN, etc

Stefan Wong 2019
"""

import os
from PIL import Image       # using PIL instead of cv2 due to torchvision
import argparse
import random
#from tqdm import tqdm
from lernomatic.data import data_split
from lernomatic.data import image_proc

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def main() -> None:
    """
    We know that the format of the dataset is a collection of sub-folders, one for each category
    of bird. For training a GAN we aren't interested in any of the labels, and so we just want to
    walk each of the subdirectories looking for image files and then flatten (and randomize) the
    heirarchy. In this case, we just place all the paths into a gigantic list and then use
    random.shuffle to select them before placing them into a dataset.
    """

    image_paths = []

    for dirname, subdir, filelist in os.walk(GLOBAL_OPTS['dataset_root']):
        if GLOBAL_OPTS['verbose']:
            print('Found directory [%s] containing %d files' % (dirname, len(filelist)))

        if len(filelist) > 0:
            for fname in filelist:
                if GLOBAL_OPTS['extension'] in fname:
                    image_paths.append(str(dirname) + '/' + str(fname))

    if GLOBAL_OPTS['verbose']:
        print('Added %d image files to dataset' % len(image_paths))

    if len(image_paths) == 0:
        raise ValueError('No image files found in directory of subdirectories of %s' %\
                         str(GLOBAL_OPTS['dataset_root'])
        )

    if GLOBAL_OPTS['randomize']:
        image_paths = random.shuffle(image_paths)

    s = data_split.DataSplit(split_name='caltech_birds')
    s.data_paths  = image_paths
    s.data_labels = [0 for x in range(len(image_paths))]
    s.elem_ids    = [0 for x in range(len(image_paths))]
    s.has_labels  = True
    s.has_ids     = True

    # process the data
    proc = image_proc.ImageDataProc(
        label_dataset_size = 1,
        image_dataset_size = (3, GLOBAL_OPTS['image_size'], GLOBAL_OPTS['image_size']),
        to_tensor = True,
        to_pil = GLOBAL_OPTS['to_pil'],
        verbose = GLOBAL_OPTS['verbose']
    )
    proc.proc(s, GLOBAL_OPTS['outfile'])


def arg_parser() -> argparse.ArgumentParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root',
                        type=str,
                        default='/mnt/ml-data/datasets/CUB_200_2011/',
                        help='Path to root of data'
                        )
    parser.add_argument('--outfile',
                        type=str,
                        default='hdf5/cub_200_2011.h5',
                        help='Filename for output HDF5 file'
                        )
    parser.add_argument('--image-size',
                        type=int,
                        default=128,
                        help='Resize all images to be this size (squared) (default: 64)'
                        )
    parser.add_argument('--extension',
                        type=str,
                        default='jpg',
                        help='Select the file extension to search for in dataset (default: jpg)'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--randomize',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--to-pil',
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

    if GLOBAL_OPTS['verbose']:
        print('---- GLOBAL OPTIONS ----')
        for k, v in GLOBAL_OPTS.items():
            print('\t [%s] : %s' % (str(k), str(v)))

    main()
