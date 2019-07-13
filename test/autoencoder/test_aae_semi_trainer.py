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
#from pudb import set_trace; set_trace()

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

        q_net = aae_common.AAEQNet(
            self.x_dim,
            self.z_dim,
            self.hidden_size,
            num_classes = self.num_classes
        )
        p_net = aae_common.AAEPNet(
            self.x_dim,
            self.z_dim+ self.num_classes,
            self.hidden_size
        )
        d_cat_net = aae_common.AAEDNetGauss(
            self.num_classes,
            self.hidden_size
        )
        d_gauss_net = aae_common.AAEDNetGauss(
            self.z_dim,
            self.hidden_size
        )

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

        self.assertIsNotNone(train_label_dataset)
        self.assertIsNotNone(train_unlabel_dataset)
        self.assertIsNotNone(val_label_dataset)

        src_trainer = aae_semisupervised_trainer.AAESemiTrainer(
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

        src_trainer.train()
        print('Saving checkpoint to file [%s]' % str(test_checkpoint_file))
        src_trainer.save_checkpoint(test_checkpoint_file)

        dst_trainer = aae_semisupervised_trainer.AAESemiTrainer(device_id = GLOBAL_OPTS['device_id'])
        self.assertEqual(None, dst_trainer.q_net)
        self.assertEqual(None, dst_trainer.p_net)
        self.assertEqual(None, dst_trainer.d_cat_net)
        self.assertEqual(None, dst_trainer.d_gauss_net)

        # Test that models, etc are loaded
        print('Loading checkpoint data from [%s]' % str(test_checkpoint_file))
        dst_trainer.load_checkpoint(test_checkpoint_file)
        self.assertIsNotNone(dst_trainer.q_net)
        self.assertIsNotNone(dst_trainer.p_net)
        self.assertIsNotNone(dst_trainer.d_cat_net)
        self.assertIsNotNone(dst_trainer.d_gauss_net)

        model_list = ['q_net', 'p_net', 'd_cat_net', 'd_gauss_net']

        for model in model_list:
            src_model = getattr(src_trainer, model)
            dst_model = getattr(dst_trainer, model)
            self.assertIsNotNone(src_model)
            self.assertIsNotNone(dst_model)
            print('\t Comparing parameters for model [%s]' % repr(src_model))
            src_model_params = src_model.get_net_state_dict()
            dst_model_params = dst_model.get_net_state_dict()

            self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

            # p1, p2 are k,v tuple pairs of each model parameters
            # k = str
            # v = torch.Tensor
            for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
                self.assertEqual(p1[0], p2[0])
                print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())), end='\r')
                self.assertEqual(True, torch.equal(p1[1], p2[1]))
            print('\n ...done')


        # Test that history is correctly loaded
        print('Saving history to file [%s]' % str(test_history_file))
        src_trainer.save_history(test_history_file)
        dst_trainer.load_history(test_history_file)

        # Check iteration values
        self.assertEqual(src_trainer.loss_iter, dst_trainer.loss_iter)
        self.assertEqual(src_trainer.val_loss_iter, dst_trainer.val_loss_iter)
        self.assertEqual(src_trainer.train_val_loss_iter, dst_trainer.train_val_loss_iter)
        self.assertEqual(src_trainer.acc_iter, dst_trainer.acc_iter)
        self.assertEqual(src_trainer.cur_epoch, dst_trainer.cur_epoch)
        self.assertEqual(src_trainer.iter_per_epoch, dst_trainer.iter_per_epoch)

        # Check history arrays
        self.assertIsNotNone(dst_trainer.d_loss_history)
        self.assertIsNotNone(dst_trainer.g_loss_history)
        self.assertIsNotNone(dst_trainer.recon_loss_history)
        self.assertIsNotNone(dst_trainer.class_loss_history)

        self.assertEqual(len(src_trainer.d_loss_history), len(dst_trainer.d_loss_history))
        self.assertEqual(len(src_trainer.g_loss_history), len(dst_trainer.g_loss_history))
        self.assertEqual(len(src_trainer.recon_loss_history), len(dst_trainer.recon_loss_history))
        self.assertEqual(len(src_trainer.class_loss_history), len(dst_trainer.class_loss_history))

        for idx in range(len(src_trainer.d_loss_history)):
            print('Checking d_loss_history idx [%d / %d]' % (idx+1, len(src_trainer.d_loss_history)), end='\r')
            self.assertEqual(src_trainer.d_loss_history[idx], dst_trainer.d_loss_history[idx])
        print('\n OK')

        for idx in range(len(src_trainer.g_loss_history)):
            print('Checking d_loss_history idx [%d / %d]' % (idx+1, len(src_trainer.g_loss_history)), end='\r')
            self.assertEqual(src_trainer.g_loss_history[idx], dst_trainer.g_loss_history[idx])
        print('\n OK')

        for idx in range(len(src_trainer.recon_loss_history)):
            print('Checking d_loss_history idx [%d / %d]' % (idx+1, len(src_trainer.recon_loss_history)), end='\r')
            self.assertEqual(src_trainer.recon_loss_history[idx], dst_trainer.recon_loss_history[idx])
        print('\n OK')

        for idx in range(len(src_trainer.class_loss_history)):
            print('Checking d_loss_history idx [%d / %d]' % (idx+1, len(src_trainer.class_loss_history)), end='\r')
            self.assertEqual(src_trainer.class_loss_history[idx], dst_trainer.class_loss_history[idx])
        print('\n OK')
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
