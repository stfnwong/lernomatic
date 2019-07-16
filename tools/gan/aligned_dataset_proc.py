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


# NOTE: the 'splits' are often pre-defined for the datasets used here, so (at
# least in the inital version) its not really worth going to a great deal of
# trouble worrying about how to divide up the splits
def main() -> None:

    test_a_root = os.path.join(GLOBAL_OPTS['data_root'], 'testA/')
    test_b_root = os.path.join(GLOBAL_OPTS['data_root'], 'testB/')

    test_a_paths = [test_a_root + path for path in os.listdir(test_a_root)]
    test_b_paths = [test_b_root + path for path in os.listdir(test_b_root)]

    print('test_a_root [%s] contains %d files' % (test_a_root, len(test_a_paths)))
    print('test_b_root [%s] contains %d files' % (test_b_root, len(test_b_paths)))




def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    # data options
    parser.add_argument('--data-root',
                        type=str,
                        default='/home/kreshnik/ml-data/monet2photo',
                        help='Path to root of dataset'
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

    #GLOBAL_OPTS['split_names'] = GLOBAL_OPTS['split_names'].split(',')
    #split_ratios = GLOBAL_OPTS['split_ratios'].split(',')
    #split_ratio_floats = []

    #for s in split_ratios:
    #    split_ratio_floats.append(float(s))

    #GLOBAL_OPTS['split_ratios'] = split_ratio_floats

    #if len(GLOBAL_OPTS['split_names']) != len(GLOBAL_OPTS['split_ratios']):
    #    raise ValueError('Number of split rations must equal number of split names')

    #if sum(split_ratio_floats) > 1.0:
    #    raise ValueError('Sum of split ratios cannot exceed 1.0')

    main()
