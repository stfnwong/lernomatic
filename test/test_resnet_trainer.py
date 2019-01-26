"""
TEST_RESNET_TRAINER
Test the resnet trainer object.

Stefan Wong 2019
"""

import sys
import argparse
import unittest
import torch
# units under test
from lernomatic.train import resnet_trainer
from lernomatic.models import resnets

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

class TestResnetTrainer(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']
        self.resnet_depth = 28
        self.test_batch_size = 64
        self.test_num_epochs = 2
        self.test_learning_rate = 0.001

    def test_save_load_checkpoint(self):
        print('======== TestResnetTrainer.save_load_checkpoint_train_test ')

        test_checkpoint_name = GLOBAL_OPTS['checkpoint_dir'] + 'resnet_trainer_checkpoint.pkl'
        # get a model
        model = resnets.WideResnet(
            self.resnet_depth,
            10,     # using CIFAR-10 data
            1
        )
        # get a traner
        src_tr = resnet_trainer.ResnetTrainer(
            model,
            # training parameters
            batch_size = self.test_batch_size,
            num_epochs = self.test_num_epochs,
            learning_rate = self.test_learning_rate,
            # device
            device_id = GLOBAL_OPTS['device_id'],
            # checkpoint
            #checkpoint_dir = GLOBAL_OPTS['checkpoint_dir'],
            #checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
            # display,
            print_every = GLOBAL_OPTS['print_every'],
            save_every = 0,
            verbose = self.verbose
        )

        if self.verbose:
            print('Created %s object' % repr(src_tr))
            print(src_tr)

        print('Training model %s for %d epochs' % (repr(src_tr), self.test_num_epochs))
        src_tr.train()

        # save the final checkpoint
        src_tr.save_checkpoint(test_checkpoint_name)

        # get a new trainer and load checkpoint
        dst_tr = resnet_trainer.ResnetTrainer(
            model
        )
        dst_tr.load_checkpoint(test_checkpoint_name)

        # Test object parameters
        self.assertEqual(src_tr.num_epochs, dst_tr.num_epochs)
        self.assertEqual(src_tr.learning_rate, dst_tr.learning_rate)
        self.assertEqual(src_tr.weight_decay, dst_tr.weight_decay)
        self.assertEqual(src_tr.print_every, dst_tr.print_every)
        self.assertEqual(src_tr.save_every, dst_tr.save_every)
        self.assertEqual(src_tr.device_id, dst_tr.device_id)

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())), end='\r')
            self.assertEqual(True, torch.equal(p1[1], p2[1]))
        print('\n ...done')

        print('======== TestResnetTrainer.save_load_checkpoint_train_test <END>')


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
