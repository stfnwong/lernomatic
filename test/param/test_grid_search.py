"""
TEST_GRID_SEARCH
Unit tests for GridSearcher

Stefan Wong 2019
"""

import sys
import argparse
import unittest

from lernomatic.options import options
# unit(s) under test
from lernomatic.param import grid_search
from lernomatic.models import common
from lernomatic.models import resnets
from lernomatic.train import cifar_trainer


GLOBAL_OPTS = dict()


# Get models, etc
def get_model() -> common.LernomaticModel:
    if GLOBAL_OPTS['model'] == 'resnet':
        model = resnets.WideResnet(
            depth=GLOBAL_OPTS['resnet_depth'],
            num_classes=10,
            input_channels = 3
        )
    elif GLOBAL_OPTS['model'] == 'cifar':
        model = cifar.CIFAR10Net()
    else:
        raise ValueError('Unknown model type [%s]' % str(GLOBAL_OPTS['model']))

    return model


def get_trainer(model, checkpoint_name:str):
    trainer = cifar_trainer.CIFAR10Trainer(
        model,
        batch_size      = GLOBAL_OPTS['batch_size'],
        val_batch_size  = GLOBAL_OPTS['val_batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        #momentum = GLOBAL_OPTS['momentum'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = checkpoint_name,
        # display,
        print_every     = GLOBAL_OPTS['print_every'],
        save_every      = GLOBAL_OPTS['save_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    return trainer



class TestGridSearcher(unittest.TestCase):
    def setUp(self):
        self.test_num_params:int = 4
        self.test_max_num_epochs:int = 24
        self.test_min_num_epochs:int = 2
        self.test_min_req_acc:float = 0.65

    def test_init(self):
        print('======== TestGridSearcher.test_init ')

        # get something to train on
        model = get_model()
        trainer = get_trainer(model, GLOBAL_OPTS['checkpoint_name'])
        # get a grid searcher
        gsearcher = grid_search.GridSearcher(
            model,
            trainer,
            num_params     = self.test_num_params,
            max_num_epochs = self.test_max_num_epochs,
            min_num_epochs = self.test_min_num_epochs,
            min_req_acc    = self.test_min_req_acc,
            verbose = True
        )

        # specify the ranges to search with a param dict
        test_params = {
            'learning_rate' : (1e-6, 1e-2),
            'weight_decay'  : (0.0, 0.2),
        }

        gsearcher.search(test_params)
        print('Created %d grid results from %d parameters' % len(gsearcher.param_history))
        gsearcher.save_history(prefix='data/')

        # TODO: make a plot of accuracy?

        for n, result in enumerate(gsearcher.param_history):
            print(n, str(result))

        print('======== TestGridSearcher.test_init <END>')



# Entry point
if __name__ == '__main__':
    parser = options.get_trainer_options()
    #parser = argparse.ArgumentParser()
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
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='grid_search_test'
                        )
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    # model type
    parser.add_argument('--model',
                        type=str,
                        default='resnet',
                        help='Type of model to use (default: resnet)'
                        )
    parser.add_argument('--resnet-depth',
                        type=int,
                        default=28,
                        help='Number of layers for Resnet model'
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
