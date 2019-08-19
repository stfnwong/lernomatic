"""
TEST_AAE_INFERRER

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
from lernomatic.infer.autoencoder import aae_inferrer


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



class TestAAEInferrer(unittest.TestCase):
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

    def test_infer(self):
        print('======== TestAAEInferrer.test_infer ')

        # get some models
        q_net = aae_common.AAEQNet(self.x_dim, self.z_dim, self.hidden_size)
        p_net = aae_common.AAEPNet(self.x_dim, self.z_dim, self.hidden_size)
        d_net = aae_common.AAEDNetGauss(self.z_dim, self.hidden_size)

        train_dataset, val_dataset = get_mnist_datasets(self.test_data_dir)

        # get a trainer
        trainer = aae_trainer.AAETrainer(
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
        # train
        trainer.train()

        # get an inferrer
        inferrer = aae_inferrer.AAEInferrer(
            q_net,
            p_net,
            device_id = GLOBAL_OPTS['device_id']
        )

        # perform inference on trained models
        for batch_idx, (data, target) in enumerate(trainer.val_loader):
            data.resize_(GLOBAL_OPTS['batch_size'], inferrer.q_net.get_x_dim())
            gen_img = inferrer.forward(data)
            self.assertIsNotNone(gen_img)
            #img_filename = 'figures/aae/aae_batch_%d.png' % int(batch_idx)
            #recon_to_plt(ax, data.cpu(), gen_img.cpu())
            #fig.savefig(img_filename, bbox_inches='tight')


        print('======== TestAAEInferrer.test_infer <END>')



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
