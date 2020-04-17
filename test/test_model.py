"""
TEST_MODEL
Unit tests for new Model arch

Stefan Wong 2019
"""

import sys
import argparse
import unittest
import importlib
import torch

from lernomatic.models import common
from lernomatic.models import cifar
from lernomatic.models import alexnet
from lernomatic.train import cifar_trainer


GLOBAL_OPTS = dict()



def get_model() -> common.LernomaticModel:
    model = cifar.CIFAR10Net()
    return model

def get_alexnet() -> common.LernomaticModel:
    model = alexnet.AlexNetCIFAR10()
    return model

def get_trainer(model : common.LernomaticModel,
                batch_size:int,
                checkpoint_name : str
                ) -> cifar_trainer.CIFAR10Trainer:
    trainer = cifar_trainer.CIFAR10Trainer(
        model,
        num_epochs = 4,
        checkpoint_name = checkpoint_name,
        # since we don't train for long a large learning rate helps
        learning_rate = 1.5e-3,
        save_every = 0,
        print_every = 50,
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

        # get a model and trainer
        model_checkpoint_file = 'checkpoint/model_save_load_test.pkl'
        src_model = get_model()
        trainer = get_trainer(
            src_model,
            64,
            checkpoint_name = 'model_save_load_test'
        )
        # train the model
        trainer.train()
        trainer.save_checkpoint(model_checkpoint_file)

        # Load data from checkpoint
        checkpoint_data = torch.load(model_checkpoint_file)
        model_import_path = checkpoint_data['model']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['model']['model_name'])
        dst_model = mod()
        dst_model.set_params(checkpoint_data['model'])
        self.assertEqual(type(src_model), type(dst_model))
        # we need this to be on the same device, which we won't get by default
        dst_model.send_to(trainer.device)

        # Check the LernomaticModel params
        src_params = src_model.get_params()
        dst_params = dst_model.get_params()

        self.assertEqual(len(src_params), len(dst_params))
        for k, v in src_params.items():
            self.assertIn(k, dst_params.keys())
            # we test the state dict in a seperate pass
            if k == 'model_state_dict':
                continue
            print('Checking [%s]' % str(k))
            self.assertEqual(v, dst_params[k])

        # Check the torch module params
        src_model_params = src_model.get_net_state_dict()
        dst_model_params = dst_model.get_net_state_dict()
        print('Comparing models')
        for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())))
            self.assertEqual(True, torch.equal(p1[1], p2[1]))
        print('\n ...done')

        print('======== TestModel.test_save_load <END>')

    def test_freeze_unfreeze(self):
        print('======== TestModel.test_freeze_unfreeze ')

        # get a pretrained model and a trainer
        unfrozen_model = get_alexnet()

        unfrozen_trainer = get_trainer(
            unfrozen_model,
            64,
            checkpoint_name = 'unfrozen_model_test'
        )
        unfrozen_trainer.train()

        # get another model and freeze all but the last two layers
        frozen_model = get_alexnet()
        #frozen_model.freeze_to(frozen_model.get_num_layers() - 1)
        # TODO: what happens when we freeze all the layers?
        frozen_model.freeze()

        frozen_trainer = get_trainer(
            unfrozen_model,
            64,
            checkpoint_name = 'frozen_model_test'
        )
        frozen_trainer.set_num_epochs(unfrozen_trainer.get_num_epochs() * 2)
        frozen_trainer.train()
        # unfreeze the model
        frozen_trainer.model.unfreeze()

        # The first N-2 layers should match the frozen model
        ref_model = get_alexnet()

        ref_model_params = ref_model.get_net_state_dict()
        frz_model_params = frozen_model.get_net_state_dict()
        print('Comparing models %s (ref), %s (frozen)' % (repr(ref_model), repr(frozen_model)))
        for n, (p1, p2) in enumerate(zip(ref_model_params.items(), frz_model_params.items())):
            self.assertEqual(p1[0], p2[0])
            if n <= frozen_model.get_num_layers() - 2:
                print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(ref_model_params.items())))
                self.assertEqual(True, torch.equal(p1[1], p2[1]))
        print('\n ...done')

        print('======== TestModel.test_freeze_unfreeze <END>')


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
