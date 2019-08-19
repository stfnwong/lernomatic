"""
TEST_ADVESARIAL_TRAINER
Unit tests for AAETrainer object

Stefan Wong 2019
"""

import sys
import argparse
import unittest
import torch
import torchvision
# module(s) under test
from lernomatic.models.autoencoder import aae_common
from lernomatic.train.autoencoder import aae_trainer


# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()


def get_mnist_datasets(data_dir:str) -> tuple:
    dataset_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize( (0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        data_dir,
        train = True,
        download = True,
        transform = dataset_transform
    )
    val_dataset = torchvision.datasets.MNIST(
        data_dir,
        train = False,
        download = True,
        transform = dataset_transform
    )

    return (train_dataset, val_dataset)


class TestAAETrainer(unittest.TestCase):
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

    def test_save_load(self):
        print('======== TestAAETrainer.test_save_load ')

        test_checkpoint_file = 'checkpoint/test_aae_trainer_checkpoint.pth'
        test_history_file = 'checkpoint/test_aae_trainer_history.pth'

        # get some models
        q_net = aae_common.AAEQNet(self.x_dim, self.z_dim, self.hidden_size)
        p_net = aae_common.AAEPNet(self.x_dim, self.z_dim, self.hidden_size)
        d_net = aae_common.AAEDNetGauss(self.z_dim, self.hidden_size)

        train_dataset, val_dataset = get_mnist_datasets(self.test_data_dir)

        # get a trainer
        src_trainer = aae_trainer.AAETrainer(
            q_net,
            p_net,
            d_net,
            # datasets
            train_dataset = train_dataset,
            val_dataset   = val_dataset,
            # train options
            num_epochs    = self.test_num_epochs,
            batch_size    = GLOBAL_OPTS['batch_size'],
            # misc
            print_every   = GLOBAL_OPTS['print_every'],
            save_every    = 0,
            device_id     = GLOBAL_OPTS['device_id'],
            verbose       = GLOBAL_OPTS['verbose']
        )
        # generate the source parameters
        src_trainer.train()
        src_trainer.save_checkpoint(test_checkpoint_file)
        src_trainer.save_history(test_history_file)

        # get a new trainer
        dst_trainer = aae_trainer.AAETrainer(device_id = GLOBAL_OPTS['device_id'])
        dst_trainer.load_checkpoint(test_checkpoint_file)

        # Check parameters of each model in turn
        src_models = [src_trainer.q_net, src_trainer.p_net, src_trainer.d_net]
        dst_models = [dst_trainer.q_net, src_trainer.p_net, src_trainer.d_net]

        for src_mod, dst_mod in zip(src_models, dst_models):

            print('\t Comparing parameters for %s model' % repr(src_mod))
            src_model_params = src_mod.get_net_state_dict()
            dst_model_params = dst_mod.get_net_state_dict()

            self.assertEqual(len(src_model_params.items()), len(dst_model_params.items()))

            # p1, p2 are k,v tuple pairs of each model parameters
            # k = str
            # v = torch.Tensor
            for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
                self.assertEqual(p1[0], p2[0])
                print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())), end='\r')
                self.assertEqual(True, torch.equal(p1[1], p2[1]))
            print('\n ...done')

        # History
        dst_trainer.load_history(test_history_file)
        self.assertIsNot(None, dst_trainer.d_loss_history)
        self.assertIsNot(None, dst_trainer.g_loss_history)
        self.assertIsNot(None, dst_trainer.recon_loss_history)

        self.assertEqual(len(src_trainer.d_loss_history), len(dst_trainer.d_loss_history))
        self.assertEqual(len(src_trainer.g_loss_history), len(dst_trainer.g_loss_history))
        self.assertEqual(len(src_trainer.recon_loss_history), len(dst_trainer.recon_loss_history))


        print('======== TestAAETrainer.test_save_load <END>')



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
