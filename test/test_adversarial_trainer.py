"""
TEST_ADVESARIAL_TRAINER
Unit tests for AdversarialTrainer object

Stefan Wong 2019
"""

import sys
import argparse
import unittest
import torchvision
# module(s) under test
from lernomatic.models.autoencoder import auto_common
from lernomatic.train.autoencoder import adversarial_trainer


# debug
from pudb import set_trace; set_trace()


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


class TestAdversarialTrainer(unittest.TestCase):
    def setUp(self):
        # MNIST sizes - unit testing on MNIST should be relatively fast
        self.num_classes   = 10
        self.hidden_size   = 1000
        self.x_dim         = 784
        self.z_dim         = 2
        self.y_dim         = 10
        self.test_data_dir = './data'
        self.test_num_epochs = 4
        self.test_batch_size = 32

    def test_save_load(self):
        print('======== TestAdversarialTrainer.test_save_load ')

        test_checkpoint_file = 'checkpoint/test_adversarial_trainer_checkpoint.pth'
        test_history_file = 'checkpoint/test_adversarial_trainer_history.pth'

        # get some models
        q_net = auto_common.AutoQNet(self.x_dim, self.z_dim, self.hidden_size)
        p_net = auto_common.AutoPNet(self.x_dim, self.z_dim, self.hidden_size)
        d_net = auto_common.AutoDNetGauss(self.z_dim, self.hidden_size)

        train_dataset, val_dataset = get_mnist_datasets(self.test_data_dir)

        # get a trainer
        trainer = adversarial_trainer.AdversarialTrainer(
            q_net,
            p_net,
            d_net,
            # datasets
            train_dataset = train_dataset,
            val_dataset = val_dataset,
            # train options
            num_epochs = self.test_num_epochs,
            batch_size = self.test_batch_size,
            # misc
            print_every = GLOBAL_OPTS['print_every'],
            save_every = 0,
            device_id = GLOBAL_OPTS['device_id'],
            verbose = GLOBAL_OPTS['verbose']
        )

        trainer.train()


        print('======== TestAdversarialTrainer.test_save_load <END>')





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
            print('\t[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()
