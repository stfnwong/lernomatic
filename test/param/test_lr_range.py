"""
TEST_LR_RANGE
Unit tests that are just for the rangefinding components of the
LRFinder class.

Stefan Wong 2019
"""

import sys
import unittest
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
# unit(s) under test
from lernomatic.param import lr_common
from lernomatic.train import cifar_trainer
from lernomatic.models import cifar
from lernomatic.vis import vis_loss_history


GLOBAL_OPTS = dict()


def get_trainer(model:common.LernomaticModel=None,
                save_every:int=0,
                print_every:int=20,
                batch_size:int=64,
                learning_rate:float=3e-4,
                num_epochs:int=10,
                device_id:int=-1,
                verbose:bool=False
                ) -> cifar_trainer.CIFAR10Trainer:
    # get a model to test on and its corresponding trainer
    model = cifar.CIFAR10Net()
    trainer = cifar_trainer.CIFAR10Trainer(
        model,
        # turn off checkpointing
        save_every = save_every,
        print_every = print_every,
        # data options
        batch_size = batch_size,
        # training options
        learning_rate = GLOBAL_TEST_PARAMS['test_learning_rate'],
        num_epochs = GLOBAL_TEST_PARAMS['train_num_epochs'],
        device_id = GLOBAL_OPTS['device_id'],
        verbose = GLOBAL_OPTS['verbose']
    )

    return trainer


def get_lr_finder(trainer, find_type:str='LogFinder') -> lr_common.LRFinder:
    if not hasattr(lr_common, find_type):
        raise ValueError('Unknown learning rate finder type [%s]' % str(find_type))

    lr_find_obj = getattr(lr_common, find_type)
    lr_finder = lr_find_obj(
        trainer,
        lr_min         = GLOBAL_OPTS['lr_min'],
        lr_max         = GLOBAL_OPTS['lr_max'],
        lr_select_method = GLOBAL_OPTS['lr_select_method'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = GLOBAL_OPTS['find_print_every']
    )

    return lr_finder



class TestLRFinderRange(unittest.TestCase):
    def test_sobel_loss(self) -> None:
        print('======== TestLRFinderRange.test_sobel_loss ')



        print('======== TestLRFinderRange.test_sobel_loss <END>')




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
                        default=64,
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
            print('\t[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()
