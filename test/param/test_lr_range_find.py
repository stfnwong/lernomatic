"""
TEST_LR_RANGE_FIND
Unit tests that are just for the rangefinding components of the LRFinder class.

Stefan Wong 2019
"""

import sys
import os
import unittest
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
# timing stuff
import time
from datetime import timedelta
# unit(s) under test
from lernomatic.param import lr_common
from lernomatic.train import cifar_trainer
from lernomatic.models import common
from lernomatic.models import cifar
from lernomatic.vis import vis_loss_history

# debug
from pudb import set_trace; set_trace()


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
        save_every    = save_every,
        print_every   = print_every,
        # data options
        batch_size    = batch_size,
        # training options
        learning_rate = learning_rate,
        num_epochs    = num_epochs,
        device_id     = device_id,
        verbose       = verbose
    )

    return trainer



def get_lr_finder(trainer,
                  find_type:str='LogFinder',
                  lr_min:float=1e-6,
                  lr_max:float=1.0,
                  lr_select_method:str='max_acc',
                  find_num_epochs:int=8,
                  find_explode_thresh:float=8.0,
                  find_print_every:int=32) -> lr_common.LRFinder:
    if not hasattr(lr_common, find_type):
        raise ValueError('Unknown learning rate finder type [%s]' % str(find_type))

    lr_find_obj = getattr(lr_common, find_type)
    lr_finder = lr_find_obj(
        trainer,
        lr_min           = lr_min,
        lr_max           = lr_max,
        lr_select_method = lr_select_method,
        num_epochs       = find_num_epochs,
        explode_thresh   = find_explode_thresh,
        print_every      = find_print_every
    )

    return lr_finder



class TestLRFinderRange(unittest.TestCase):
    def test_kde_loss(self) -> None:
        print('======== TestLRFinderRange.test_kde_loss ')

        test_finder_history = 'checkpoint/lr_find_sobel_loss.pth'

        # Just use a simple CIFAR10 network for this
        model = cifar.CIFAR10Net()
        trainer = get_trainer(
            model,
            device_id  = GLOBAL_OPTS['device_id'],
            batch_size = GLOBAL_OPTS['batch_size']
        )

        # defaults are fine here
        if os.path.exists(test_finder_history):
            lr_finder = lr_common.lr_finder_auto_load(test_finder_history)
            lr_finder.trainer = trainer
        else:
            lr_finder = get_lr_finder(trainer, lr_select_method='kde')

        find_start_time = time.time()
        lr_find_min, lr_find_max = lr_finder.find()
        find_end_time = time.time()
        find_total_time = find_end_time - find_start_time

        lr_finder.save(test_finder_history)

        print('Acc history contains %d elements' % len(lr_finder.acc_history))
        print('Found learning rate range %.4f -> %.4f in %s' %\
              (lr_find_min, lr_find_max, str(timedelta(seconds = find_total_time)))
        )

        fig, ax = plt.subplots()
        lr_finder.plot_lr_vs_acc(ax)
        #ax.plot(np.asarray(lr_finder.acc_history))
        #ax.axvline(x = lr_find_min, color='r')
        #ax.axvline(x = lr_find_max, color='r')

        fig.tight_layout()
        fig.savefig('lr_find_max_acc.png')

        print('======== TestLRFinderRange.test_kde_loss <END>')


    def test_max_acc(self):
        print('======== TestLRFinderRange.test_max_acc ')

        test_finder_history = 'checkpoint/lr_max_acc_loss.pth'

        # Just use a simple CIFAR10 network for this
        model = cifar.CIFAR10Net()
        trainer = get_trainer(
            model,
            device_id  = GLOBAL_OPTS['device_id'],
            batch_size = GLOBAL_OPTS['batch_size']
        )

        # defaults are fine here
        if os.path.exists(test_finder_history):
            lr_finder = lr_common.lr_finder_auto_load(test_finder_history)
            lr_finder.trainer = trainer
        else:
            lr_finder = get_lr_finder(trainer, lr_select_method='kde')

        find_start_time = time.time()
        lr_find_min, lr_find_max = lr_finder.find()
        find_end_time = time.time()
        find_total_time = find_end_time - find_start_time

        lr_finder.save(test_finder_history)

        print('Acc history contains %d elements' % len(lr_finder.acc_history))
        print('Found learning rate range %.4f -> %.4f in %s' %\
              (lr_find_min, lr_find_max, str(timedelta(seconds = find_total_time)))
        )

        fig, ax = plt.subplots()
        lr_finder.plot_lr_vs_acc(ax)
        #ax.plot(np.asarray(lr_finder.acc_history))
        #ax.axvline(x = lr_find_min, color='r')
        #ax.axvline(x = lr_find_max, color='r')

        fig.tight_layout()
        fig.savefig('lr_find_kde.png')

        print('======== TestLRFinderRange.test_max_acc <END>')



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
