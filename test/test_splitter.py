"""
TEST_SPLITTER
Unit tests for data splitter

Stefan Wong 2019
"""

import os
import sys
import unittest
import argparse
# units under test
from lernomatic.data import split


# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


class TestListSplitter(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']
        self.test_split_ratios = [0.7, 0.2, 0.1]
        self.test_split_names = ['ut_test', 'ut_train', 'ut_val']
        self.test_split_method = 'random'
        #self.test_split_root = GLOBAL_OPTS['test_data_root']

    # TODO : test exceptions in constructor

    def test_gen_split(self):
        print('======== TestListSplitter.test_gen_split ')

        splitter = split.ListSplitter(
            split_ratios    = self.test_split_ratios,
            split_names     = self.test_split_names,
            split_method    = self.test_split_method,
            #split_data_root = self.test_split_root,
            split_data_root = GLOBAL_OPTS['test_data_root'],
            verbose         = self.verbose
        )
        print('Generated new splitter object')
        print(splitter)

        print('Generating data list....', end=' ')
        cat_list = os.listdir(GLOBAL_OPTS['test_data_root'] + '/Cat')
        dog_list = os.listdir(GLOBAL_OPTS['test_data_root'] + '/Dog')
        print('done')

        print('Generating label list....', end=' ')
        label_list = [0] * len(cat_list) +\
                     [1]* len(dog_list)
        print('done')

        print('Splitting...')
        splits = splitter.gen_splits(
            cat_list + dog_list,
            label_list
        )

        print('done\n')

        for n, s in enumerate(splits):
            print('Split %d <%s> : length : %d' % (n+1, s.get_name(), len(s)))

        print('======== TestListSplitter.test_gen_split <END>')



# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--test-data-root',
                        type=str,
                        default='/mnt/ml-data/datasets/cats-vs-dogs/PetImages',
                        help='Root of path to image data'
                        )

    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    arg_vals = vars(args)

    print(arg_vals.keys())

    for k, v in arg_vals.items():
        print('%s : %s' % (str(k), str(v)))
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print('-------- GLOBAL OPTS (%s) --------' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('[%s] : %s' % (str(k), str(v)))


    sys.argv[1:] = args.unittest_args
    unittest.main()
