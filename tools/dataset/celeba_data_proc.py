"""
CELEBA_DATA_PROC
Create CelebA HDF5 dataset files

Stefan Wong 2019
"""

import os
import cv2
import argparse
#from tqdm import tqdm
from lernomatic.data import data_split
from lernomatic.data import image_proc

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def main():

    img_paths = os.listdir(GLOBAL_OPTS['dataset_root'])
    src_len = len(img_paths)
    if img_paths is None or len(img_paths) < 1:
        raise ValueError('No files in directory [%s]' % GLOBAL_OPTS['dataset_root'])

    print('Checking files in path [%s]' % GLOBAL_OPTS['dataset_root'])
    num_err = 0
    for n, path in enumerate(img_paths):
        img = cv2.imread(path)
        if img is None:
            print('Failed to load image [%d/%d] <%s>' % (n, len(img_paths), str(path)), end='\r')
            img_paths.pop(n)
            num_err += 1

    print('\nChecked %d images, %d failed to load (%d remaining)' % (src_len, num_err, len(img_paths)))

    # Don't need to create splits, so a single split object is sufficient
    s = data_split.DataSplit(split_name='celeba')
    s.data_paths  = img_paths
    s.data_labels = [0 for x in range(len(img_paths))]
    s.elem_ids    = [0 for x in range(len(img_paths))]

    # process the data
    # TODO : option to change size
    proc = image_proc.ImageDataProc(
        label_dataset_size = 1,
        image_dataset_size = (3, GLOBAL_OPTS['img_size'], GLOBAL_OPTS['img_size']),
        verbose = GLOBAL_OPTS['verbose']
    )
    proc.proc(s, GLOBAL_OPTS['outfile'])



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root',
                        type=str,
                        default='/mnt/ml-data/datasets/celeba/img_align_celeba/',
                        help='Path to root of data'
                        )
    parser.add_argument('--outfile',
                        type=str,
                        default='hdf5/celeba.h5',
                        help='Filename for output HDF5 file'
                        )
    parser.add_argument('--img-size',
                        type=int,
                        default=64,
                        help='Resize all images to be this size (squared) (default: 64)'
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

    if GLOBAL_OPTS['verbose']:
        print('---- GLOBAL OPTIONS ----')
        for k, v in GLOBAL_OPTS.items():
            print('\t [%s] : %s' % (str(k), str(v)))

    main()
