"""
TEST_LMDB_DATASET
Unit tests for LMDB dataset object

Stefan Wong 2019
"""


import os
import sys
import PIL
import unittest
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

# unit(s) under test
from lernomatic.data import lmdb_dataset

# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()


class TestLMDBDataset(unittest.TestCase):

    def setUp(self):
        self.test_lmdb_root = '/home/kreshnik/ml-data/dining_room_train_lmdb/'

    def test_init(self):
        print('======== TestLMDBDataset.test_init ')

        dataset = lmdb_dataset.LMDBDataset(
            self.test_lmdb_root
        )

        print('dataset contains %d items' % len(dataset))

        for n, (image, target) in enumerate(dataset):
            print('Checking element [%d / %d]' % (n+1, len(dataset)), end='\r')
            self.assertEqual(True, isinstance(image, PIL.Image.Image))
            self.assertEqual(0, target)          # TODO : change when loader has changed

            #print('Element %d : [%s] : %s' % (n, type(image), type(target)))

        print('\n OK')

        print('======== TestLMDBDataset.test_init <END>')



# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--draw-plot',
                        action='store_true',
                        default=False,
                        help='Draw plots'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of worker processes to use for HDF5 load'
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=8,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Device to use for tests (default : -1)'
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
