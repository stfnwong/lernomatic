"""
TEST_MODEL
Unit tests for new Model arch

Stefan Wong 2019
"""

import sys
import argparse
import unittest

from lernomatic.models import cifar
from lernomatic.train import cifar_trainer

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def get_model() -> common.LernomaticModel:
    model = cifar.CIFAR10Net()
    return model

def get_trainer(model : common.LernomaticModel
                checkpoint_name : str,
                batch_size:int,
                ) -> cifar_trainer.CIFAR10Trainer:
    trainer = cifar_trainer.CIFART10rainer(
        model,
        num_epochs = 4,
        checkpoint_name = checkpoint_name,
        batch_size = batch_size,
        device_id = GLOBAL_OPTS['device_id'],
        verbose = GLOBAL_OPTS['verbose']
    )
    return trainer


class TestModel(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']

    def test_save_load(self):
        print('======== TestModel.test_save_load ')

        model = get_model()
        trainer = get_trainer(
            model,
            checkpoint_name = 'model_save_load_test'
        )
        # train the model
        trainer.train()

        print('======== TestModel.test_save_load <END>')


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
    # display options
    parser.add_argument('--print-every',
                        type=int,
                        default=100,
                        help='Print output every time this number of iterations has elapsed'
                        )
    # checkpoint options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='checkpoint/',
                        help='Directory to save checkpoints to'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='resnet-trainer-test',
                        help='String to prefix to checkpoint files'
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
