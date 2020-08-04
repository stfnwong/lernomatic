"""
ALIGNED_DATASET_PROC
Processed an algined image dataset

Stefan Wong 2019
"""

import os
import argparse
from lernomatic.data.gan import aligned_dataset
from lernomatic.data.gan import aligned_data_proc
from lernomatic.util import file_util

GLOBAL_OPTS = dict()

VALID_TOOL_MODES = ('join', 'split')


# debug
#


def split() -> None:

    data_paths = file_util.get_file_paths(GLOBAL_OPTS['dataset_root'], verbose=GLOBAL_OPTS['verbose'])
    if len(data_paths) == 0:
        raise ValueError('Failed to find any images in path or subpaths of [%s]' % GLOBAL_OPTS['dataset_root'])

    if GLOBAL_OPTS['size'] > 0:
        data_paths = data_paths[0 : GLOBAL_OPTS['size']]

    # Randomize order?
    image_shape = (GLOBAL_OPTS['num_channels'], GLOBAL_OPTS['image_size'], GLOBAL_OPTS['image_size'])

    data_proc = aligned_data_proc.AlignedImageSplit(
        a_img_name = GLOBAL_OPTS['a_img_name'],
        b_img_name = GLOBAL_OPTS['b_img_name'],
        image_shape = image_shape,
        verbose = GLOBAL_OPTS['verbose']
    )
    data_proc.proc(data_paths, GLOBAL_OPTS['outfile'])



def join() -> None:
    test_a_paths = [test_a_root + path for path in os.listdir(GLOBAL_OPTS['a_data_root'])]
    test_b_paths = [test_b_root + path for path in os.listdir(GLOBAL_OPTS['b_data_root'])]

    if GLOBAL_OPTS['size'] > 0:
        test_a_paths = test_a_paths[0 : GLOBAL_OPTS['size']]
        test_b_paths = test_b_paths[0 : GLOBAL_OPTS['size']]

    print('A data root [%s] contains %d files' % (GLOBAL_OPTS['a_data_root'], len(test_a_paths)))
    print('B data root [%s] contains %d files' % (GLOBAL_OPTS['b_data_root'], len(test_b_paths)))

    image_shape = (GLOBAL_OPTS['num_channels'], GLOBAL_OPTS['image_size'], GLOBAL_OPTS['image_size'])
    data_proc = aligned_data_proc.AlignedImageJoin(
        image_size = image_size,
        verbose = GLOBAL_OPTS['verbose']
    )

    if GLOBAL_OPTS['verbose']:
        print('Generated new %s object' % repr(data_proc))

    data_proc.proc(test_a_paths, test_b_paths, GLOBAL_OPTS['dataset_outfile'])



def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--mode',
                        type=str,
                        choices=VALID_TOOL_MODES,
                        default='split',
                        help='Tool mode (default: split)'
                        )
    # data options
    parser.add_argument('--outfile',
                        type=str,
                        default='./hdf5/night2day_aligned.h5',
                        help='Name of ouput data file'
                        )
    parser.add_argument('--dataset-root',
                        type=str,
                        default='/home/kreshnik/ml-data/night2day',
                        help='Root to path of AB images (images that have already been aligned)'
                        )
    parser.add_argument('--a-data-root',
                        type=str,
                        default=None,
                        help='(In join mode) Path to A images'
                        )
    parser.add_argument('--b-data-root',
                        type=str,
                        default=None,
                        help='(In join mode) Path to B images'
                        )
    parser.add_argument('--extension',
                        type=str,
                        default='jpg',
                        help='Select the file extension to search for in dataset (default: jpg)'
                        )
    parser.add_argument('--randomize',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    # Image options
    parser.add_argument('--image-size',
                        type=int,
                        default=256,
                        help='Single dimension of one image in dataset. Images are croppped to be square (default: 256)'
                        )
    parser.add_argument('--num-channels',
                        type=int,
                        default=3,
                        help='Number of channels in output image (default: 3)'
                        )
    # Dataset options
    parser.add_argument('--a-img-name',
                        type=str,
                        default='a_imgs',
                        help='Name of A image dataset (default: a_imgs)'
                        )
    parser.add_argument('--b-img-name',
                        type=str,
                        default='b_imgs',
                        help='Name of B image dataset (default: b_imgs)'
                        )
    parser.add_argument('--size',
                        type=int,
                        default=0,
                        help='Max size of dataset. If zero, then all files found are used (default: 0)'
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
            print('\t[%s] : %s' % (str(k), str(v)))

    if GLOBAL_OPTS['mode'] == 'split':
        split()
    elif GLOBAL_OPTS['mode'] == 'join':
        join()
    else:
        raise ValueError('Invalid mode [%s]' % str(GLOBAL_OPTS['mode']))
