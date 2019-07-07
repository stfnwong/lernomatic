"""
TEST_AAE_SEMI_TRAINER
Unit tests for Semi-supervised Adversarial Autoencoder Trainer

Stefan Wong 2019
"""

import sys
import argparse
import unittest
import torch
import torchvision
# module(s) under test
from lernomatic.models.autoencoder import aae_common
from lernomatic.train.autoencoder import aae_semisupervised_trainer
from lernomatic.data.mnist import mnist_sub

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


class TestAAESemiTrainer(unittest.TestCase):
    def setUp(self):
        # MNIST sizes - unit testing on MNIST should be relatively fast
        self.num_classes     = 10
        self.hidden_size     = 1000
        self.x_dim           = 784
        self.z_dim           = 2
        self.y_dim           = 10
        self.test_data_dir   = './data'
        self.test_num_epochs = 4
        self.test_batch_size = GLOBAL_OPTS['batch_size']
        self.transform       = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize( (0.1307,), (0.3081,))
        ])

    def test_save_load(self):
        print('======== TestAAESemiTrainer.test_save_load ')

        test_checkpoint_file = 'checkpoint/test_aae_semi_trainer_checkpoint.pth'
        test_history_file = 'checkpoint/test_aae_semi_trainer_history.pth'

        q_net       = aae_common.AAEQNet(self.x_dim, self.z_dim, self.hidden_size)
        p_net       = aae_common.AAEPNet(self.x_dim, self.z_dim, self.hidden_size)
        d_cat_net   = aae_common.AAEDNetGauss(self.z_dim, self.hidden_size)
        d_gauss_net = aae_common.AAEDNetGauss(self.z_dim, self.hidden_size)

        q_net.set_cat_mode()
        self.assertEqual(True, q_net.net.cat_mode)

        # We also need to sub-sample some parts of the MNIST dataset to produce the
        # 'labelled' data loaders
        print('Creating MNIST sub-dataset...')
        train_label_dataset, val_label_dataset, train_unlabel_dataset = mnist_sub.gen_mnist_subset(
            self.test_data_dir,
            transform = self.transform,
            verbose = GLOBAL_OPTS['verbose']
        )

        trainer = aae_semisupervised_trainer.AAESemiTrainer(
            q_net,
            p_net,
            d_cat_net,
            d_gauss_net,
            # datasets
            train_label_dataset   = train_label_dataset,
            train_unlabel_dataset = train_unlabel_dataset,
            val_label_dataset     = val_label_dataset,
            # train options
            num_epochs    = self.test_num_epochs,
            batch_size    = GLOBAL_OPTS['batch_size'],
            # misc
            print_every   = GLOBAL_OPTS['print_every'],
            save_every    = 0,
            device_id     = GLOBAL_OPTS['device_id'],
            verbose       = GLOBAL_OPTS['verbose']
        )

        trainer.train()


        print('======== TestAAESemiTrainer.test_save_load <END>')


# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    # display
    parser.add_argument('--print-every',
                        type=int,
                        default=10,
                        help='How often to print trainer output (default: 10)'
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
