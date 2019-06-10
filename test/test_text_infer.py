"""
TEST_TEXT_INFER

Stefan Wong 2019
"""
import sys
import argparse
import unittest
import torch
import h5py
import numpy as np
from tqdm import tqdm
# modules under test
from lernomatic.model.text import seq2seq
from lernomatic.infer.text import greedy_decoder

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()



# TODO : need new (simpler) models and data to make the unit test tractable
class TestGreedyDecoder(unittest.TestCase):


        print('======== TestGreedyDecoder.test_search <END>')





# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--data-root',
                        type=str,
                        default='/mnt/ml-data/datasets/',
                        help='Path to root of dataset'
                        )

    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    arg_vals = vars(args)
    for k, v in arg_vals.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print('-------- GLOBAL OPTS (%s) --------' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()
