"""
CVD_DATA_PREP
Another script to prepare data for Cats-vs-Dogs

"""

import os
import re
import shutil
import argparse


GLOBAL_OPTS = dict()


def main():

    if GLOBAL_OPTS['train_dir'] is None:
        raise ValueError('No train directory supplied')
    if GLOBAL_OPTS['test_dir'] is None:
        raise ValueError('No test directory supplied')

    train_dogs_dir = GLOBAL_OPTS['train_dir'] + 'dogs'
    train_cats_dir = GLOBAL_OPTS['train_dir'] + 'cats'
    test_dogs_dir  = GLOBAL_OPTS['test_dir']  + 'dogs'
    test_cats_dir  = GLOBAL_OPTS['test_dir']  + 'cats'


    train_files = os.listdir(GLOBAL_OPTS['train_dir'])
    for n, f in enumerate(train_files):
        print('Checking file <%s> [%d / %d]' % (str(f), n+1, len(train_files)), end='\r')
        cat_search_obj = re.search("cat", f)
        dog_search_obj = re.search("dog", f)
        if cat_search_obj:
            shutil.move(GLOBAL_OPTS['train_dir'] + '/%s' % f, train_cats_dir)
        elif dog_search_obj:
            shutil.move(GLOBAL_OPTS['train_dir'] + '/%s' % f, train_dogs_dir)

    print('\n ...done')

    # Move some dog images to test set
    test_dog_files = os.listdir(train_dogs_dir)
    for n, f in enumerate(test_dog_files):
        print('Generating dog eval images [%d / %d]' % (n+1, GLOBAL_OPTS['num_test']), end='\r')
        if n == GLOBAL_OPTS['num_test']-1:
            break
        shutil.move(train_dogs_dir + '/%s' % f, test_dogs_dir)
    print('\n ...done')

    # Move some cat images to test set
    test_cat_files = os.listdir(train_cats_dir)
    for n, f in enumerate(test_cat_files):
        print('Generating cat eval images [%d / %d]' % (n+1, GLOBAL_OPTS['num_test']), end='\r')
        if n == GLOBAL_OPTS['num_test']-1:
            break
        shutil.move(train_cats_dir + '/%s' % f, test_cats_dir)
    print('\n ...done')



def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--train-dir',
                        type=str,
                        default=None,
                        help='Path to train directory'
                        )
    parser.add_argument('--test-dir',
                        type=str,
                        default=None,
                        help='Path to test directory'
                        )
    parser.add_argument('--num-test',
                        type=int,
                        default=1000,
                        help='Number of images to include in test data'
                        )
    # Data source options
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
