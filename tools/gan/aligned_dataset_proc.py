"""
ALIGNED_DATASET_PROC
Processed an algined image dataset

Stefan Wong 2019
"""

import os
import argparse
from lernomatic.data.gan import aligned_dataset
from lernomatic.data.gan import aligned_data_proc

GLOBAL_OPTS = dict()
VALID_TOOL_MODES = ('generate', 'load')


# NOTE: the 'splits' are often pre-defined for the datasets used here, so (at
# least in the inital version) its not really worth going to a great deal of
# trouble worrying about how to divide up the splits
def generate() -> None:

    test_a_root = os.path.join(GLOBAL_OPTS['data_root'], 'testA/')
    test_b_root = os.path.join(GLOBAL_OPTS['data_root'], 'testB/')

    test_a_paths = [test_a_root + path for path in os.listdir(test_a_root)]
    test_b_paths = [test_b_root + path for path in os.listdir(test_b_root)]

    print('test_a_root [%s] contains %d files' % (test_a_root, len(test_a_paths)))
    print('test_b_root [%s] contains %d files' % (test_b_root, len(test_b_paths)))

    data_proc = aligned_data_proc.AlignedImageProc(
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
                        default='generate',
                        help='Select tool mode from one of [%s] (default: generate)' % str(VALID_TOOL_MODES)
                        )
    # data options
    parser.add_argument('--data-root',
                        type=str,
                        default='/home/kreshnik/ml-data/night2day',
                        help='Path to root of dataset'
                        )
    parser.add_argument('--dataset-outfile',
                        type=str,
                        default='./hdf5/night2day_aligned.h5',
                        help='Name of ouput data file'
                        )
    parser.add_argument('--ab-root',
                        type=str,
                        default='/home/kreshnik/ml-data/night2day',
                        help='Root to path of AB images (images that have already been aligned)'
                        )
    # Image options
    parser.add_argument('--img-size',
                        type=int,
                        default=256,
                        help='Single dimension of one image in dataset. Images are croppped to be square (default: 256)'
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

    if GLOBAL_OPTS['mode'] == 'generate':
        generate()
    elif GLOBAL_OPTS['mode'] == 'load':
        load()
    else:
        raise ValueError('Invalid mode [%s]' % str(GLOBAL_OPTS['mode']))
