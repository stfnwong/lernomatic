"""
TEST_LEARNING_RATE
Unit test for learning rate finder

Stefan Wong 2019
"""

import sys
import unittest
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
# unit(s) under test
from lernomatic.param import learning_rate
from lernomatic.train import cifar10_trainer
from lernomatic.models import cifar10

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

class TestLRFinder(unittest.TestCase):
    def setUp(self):
        self.verbose            = GLOBAL_OPTS['verbose']
        self.test_batch_size    = 8
        self.test_learning_rate = 0.001
        self.test_print_every   = 20
        # options for learning rate finder
        self.test_start_lr      = 1e-7
        self.test_end_lr        = 1
        self.test_num_iter      = 5000

    def get_trainer(self):
        # get a model to test on and its corresponding trainer
        model = cifar10.CIFAR10Net()
        trainer = cifar10_trainer.CIFAR10Trainer(
            model,
            # turn off checkpointing
            save_every = 0,
            print_every = self.test_print_every,
            # data options
            batch_size = self.test_batch_size,
            # training options
            learning_rate = self.test_learning_rate,
            device_id = GLOBAL_OPTS['device_id'],
            verbose = self.verbose
        )

        return trainer

    def test_find_lr(self):
        print('======== TestLRFinder.test_find_lr ')

        # get an LRFinder
        trainer = self.get_trainer()
        lr_finder = learning_rate.LRFinder(
            trainer,
            start_lr = self.test_start_lr,
            end_lr   = self.test_end_lr,
            num_iter = self.test_num_iter,
            verbose  = self.verbose
        )

        if self.verbose:
            print('Created LRFinder object')
            print(lr_finder)

        lr_finder.find_lr(
            print_every = int(self.test_num_iter / 2)
        )


        print('======== TestLRFinder.test_find_lr <END>')

    def test_lr_range_find(self):
        print('======== TestLRFinder.test_lr_range_find ')

        trainer = self.get_trainer()
        lr_finder = learning_rate.LRFinder(
            trainer,
            start_lr = self.test_start_lr,
            end_lr   = self.test_end_lr,
            num_iter = self.test_num_iter,
            verbose  = self.verbose
        )

        # shut linter up
        if self.verbose:
            print(lr_finder)

        print('======== TestLRFinder.test_lr_range_find <END>')


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
                        default=0,
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
            print('[%s] : %s' % (str(k), str(v)))


    sys.argv[1:] = args.unittest_args
    unittest.main()
