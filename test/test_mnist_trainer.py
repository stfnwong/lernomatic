"""
TEST_MNIST_TRAINER
Unit tests for MNIST trainer object

Stefan Wong 2018
"""

import sys
import unittest
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
# unit(s) under test
from lernomatic.train import mnist_trainer
from lernomatic.models import mnist as mnist_net

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

class TestMNISTTrainer(unittest.TestCase):
    def setUp(self):
        self.verbose         = GLOBAL_OPTS['verbose']
        self.draw_plot       = GLOBAL_OPTS['draw_plot']
        self.test_batch_size = 16

    def test_save_load_checkpoint(self):
        print('======== TestMNISTTrainer.test_save_load_checkpoint ')

        test_dataset_file = 'hdf5/trainer_unit_test.h5'
        model = mnist_net.MNISTNet()

        test_num_epochs = 1
        src_tr = mnist_trainer.MNISTTrainer(
            model,
            num_epochs = test_num_epochs,
            save_every = 1,
            device_id = GLOBAL_OPTS['device_id'],
            # dataload options
            checkpoint_name = 'save_load_test',
            batch_size = 16,
            num_workers = GLOBAL_OPTS['num_workers']
        )

        if self.verbose:
            print('Created trainer object')
            print(src_tr)

        # train for one epoch
        src_tr.train()
        # Make a new trainer and load all parameters into that
        # I guess we need to put some kind of loader and model here...
        dst_tr = mnist_trainer.MNISTTrainer(
            model,
            device_id = GLOBAL_OPTS['device_id']
        )
        dst_tr.load_checkpoint('checkpoint/save_load_test_epoch-0.pkl')

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

        # Test loss history
        print('\t Comparing loss history....')
        self.assertEqual(src_tr.loss_iter, dst_tr.loss_iter)
        for n in range(src_tr.loss_iter):
            print('Checking loss element [%d/%d]' % (n, src_tr.loss_iter), end='\r')
            self.assertEqual(src_tr.loss_history[n], dst_tr.loss_history[n])

        print('\n ...done')

        print('======== TestMNISTTrainer.test_save_load_checkpoint <END>')

    def test_save_load_acc(self):
        print('======== TestMNISTTrainer.test_save_load_acc ')

        test_dataset_file = 'hdf5/trainer_unit_test.h5'
        val_dataset_file = 'hdf5/warblr_data.h5'
        model = mnist_net.MNISTNet()

        # Get trainer object
        test_num_epochs = 10
        src_tr = mnist_trainer.MNISTTrainer(
            model,
            save_every = 1,
            print_every = 50,
            checkpoint_name = 'save_load_test',
            device_id = GLOBAL_OPTS['device_id'],
            # loader options,
            num_epochs = test_num_epochs,
            batch_size = GLOBAL_OPTS['batch_size'],
            num_workers = GLOBAL_OPTS['num_workers'],
            val_data_path = val_dataset_file,
        )

        if self.verbose:
            print('Created trainer object')
            print(src_tr)

        # train for one epoch
        src_tr.train()
        self.assertIsNot(None, src_tr.acc_history)

        # Now try to load a checkpoint and ensure that there is an
        # acc history attribute that is not None
        # TODO : check that we restore the loaders  as well
        dst_tr = mnist_trainer.MNISTTrainer(
            model,
            device_id = GLOBAL_OPTS['device_id']        # TODO : not in checkpoint data...
        )
        ck_fname = 'checkpoint/save_load_test_epoch-%d.pkl' % (test_num_epochs-1)
        dst_tr.load_checkpoint(ck_fname)
        #dst_tr.load_checkpoint('checkpoint/save_load_test_epoch-%d.pkl' % test_num_epochs-1)
        self.assertIsNot(None, dst_tr.acc_history)

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

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n, len(src_model_params.items())), end='\r')
            self.assertEqual(True, torch.equal(p1[1], p2[1]))
        print('\n ...done')

        # Test loss history
        print('\t Comparing loss history....')
        self.assertEqual(src_tr.loss_iter, dst_tr.loss_iter)
        for n in range(src_tr.loss_iter):
            print('Checking loss element [%d/%d]' % (n, src_tr.loss_iter), end='\r')
            self.assertEqual(src_tr.loss_history[n], dst_tr.loss_history[n])

        print('======== TestMNISTTrainer.test_save_load_acc <END>')



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
            print('[%s] : %s' % (str(k), str(v)))


    sys.argv[1:] = args.unittest_args
    unittest.main()
