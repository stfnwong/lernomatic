""""
TEST_INFERRER
Unit tests for Inferrer module

Stefan Wong 2019
"""

import sys
import argparse
import unittest
import torch
# unit(s) under test
from lernomatic.models import common
from lernomatic.models import cifar
from lernomatic.train import cifar_trainer
from lernomatic.infer import inferrer

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def get_model() -> common.LernomaticModel:
    model = cifar.CIFAR10Net()
    return model

def get_trainer(model:common.LernomaticModel,
                checkpoint_name:str,
                batch_size:int,
                save_every:int) -> cifar_trainer.CIFAR10Trainer:
    trainer = cifar_trainer.CIFAR10Trainer(
        model,
        batch_size = batch_size,
        test_batch_size = 1,
        device_id = GLOBAL_OPTS['device_id'],
        checkpoint_name = checkpoint_name,
        save_every = save_every,
        save_hist = False,
        print_every = 50,
        num_epochs = 4,
        learning_rate = 9e-4
    )

    return trainer

class TestInferrer(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']

    def test_save_load(self):
        print('======== TestInferrer.test_save_load ')

        infer_test_checkpoint = 'checkpoint/infer_save_load_test.pkl'

        model = get_model()
        trainer = get_trainer(model, None, 64, 0)
        # train the model for a while
        trainer.train()
        # save a training checkpoint to disk and load it into an inferrer
        trainer.save_checkpoint(infer_test_checkpoint)

        infer = inferrer.Inferrer(device_id = GLOBAL_OPTS['device_id'])
        infer.load_model(infer_test_checkpoint)

        infer_model = infer.get_model()
        trainer_model = trainer.get_model()

        # check model parameters
        train_model_params = trainer.model.get_net_state_dict()
        infer_model_params = infer.model.get_net_state_dict()
        print('Comparing models')
        for n, (p1, p2) in enumerate(zip(train_model_params.items(), infer_model_params.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(train_model_params.items())))
            self.assertEqual(True, torch.equal(p1[1], p2[1]))
        print('\n ...done')

        # run the forward pass
        test_img, _ = next(iter(trainer.test_loader))
        pred = infer.forward(test_img)
        print('Complete prediction vector (shape: %s)' % (str(pred.shape)))
        print(str(pred))

        print('======== TestInferrer.test_save_load <END>')


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
                        help='Number of worker processes to use for reading HDF5'
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
            print('[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()


