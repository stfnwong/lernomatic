"""
TEST_DAE_TRAINER
Unit tests for Denoising Autoencoder Trainer

Stefan Wong 2019
"""

import sys
import argparse
import unittest
import torch
import torchvision
import time
from datetime import timedelta

from lernomatic.models.autoencoder import denoise_ae
from lernomatic.train.autoencoder import dae_trainer


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


class TestDAETrainer(unittest.TestCase):
    def setUp(self) -> None:
        self.verbose = GLOBAL_OPTS['verbose']
        self.test_num_epochs = 4

    def test_save_load(self) -> None:
        print('======== TestDAETrainer.test_save_load ')

        train_dataset, val_dataset = get_mnist_datasets(GLOBAL_OPTS['data_dir'])

        # Get some models. For this test we just accept the default constructor
        # parameters (num_blocks = 4, start_size = 32, kernel_size = 3)
        encoder = denoise_ae.DAEEncoder()
        decoder = denoise_ae.DAEDecoder()

        test_checkpoint_file = 'checkpoint/dae_trainer_checkpoint.pkl'
        test_history_file    = 'checkpoint/dae_trainer_history.pkl'
        src_trainer = dae_trainer.DAETrainer(
            encoder,
            decoder,
            # datasets
            train_dataset = train_dataset,
            val_dataset   = val_dataset,
            device_id     = GLOBAL_OPTS['device_id'],
            # trainer params
            batch_size = GLOBAL_OPTS['batch_size'],
            num_epochs = self.test_num_epochs,
            # disable saving
            save_every = 0,
            print_every = GLOBAL_OPTS['print_every'],
            verbose = self.verbose
        )
        train_start_time = time.time()
        src_trainer.train()
        train_end_time = time.time()
        train_total_time = train_end_time - train_start_time

        print('Trainer %s trained %d epochs in %s' %\
                (repr(self), src_trainer.cur_epoch, str(timedelta(seconds = train_total_time)))
        )

        print('Saving checkpoint to file [%s]' % str(test_checkpoint_file))
        src_trainer.save_checkpoint(test_checkpoint_file)
        src_trainer.save_history(test_history_file)

        # get a new trainer and load
        dst_trainer = dae_trainer.DAETrainer(device_id = GLOBAL_OPTS['device_id'])
        dst_trainer.load_checkpoint(test_checkpoint_file)

        # check the basic trainer params
        self.assertEqual(src_trainer.num_epochs, dst_trainer.num_epochs)
        self.assertEqual(src_trainer.learning_rate, dst_trainer.learning_rate)
        self.assertEqual(src_trainer.momentum, dst_trainer.momentum)
        self.assertEqual(src_trainer.weight_decay, dst_trainer.weight_decay)
        self.assertEqual(src_trainer.loss_function, dst_trainer.loss_function)
        self.assertEqual(src_trainer.optim_function, dst_trainer.optim_function)
        self.assertEqual(src_trainer.cur_epoch, dst_trainer.cur_epoch)
        self.assertEqual(src_trainer.iter_per_epoch, dst_trainer.iter_per_epoch)
        self.assertEqual(src_trainer.save_every, dst_trainer.save_every)
        self.assertEqual(src_trainer.print_every, dst_trainer.print_every)
        self.assertEqual(src_trainer.batch_size, dst_trainer.batch_size)
        self.assertEqual(src_trainer.val_batch_size, dst_trainer.val_batch_size)
        self.assertEqual(src_trainer.shuffle, dst_trainer.shuffle)

        # Now check the models
        src_models = [src_trainer.encoder, src_trainer.decoder]
        dst_models = [dst_trainer.encoder, dst_trainer.decoder]

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

        print('Checking history...')
        dst_trainer.load_history(test_history_file)

        self.assertEqual(len(src_trainer.loss_history), len(dst_trainer.loss_history))
        for elem in range(len(src_trainer.loss_history)):
            print('Checking loss history element [%d / %d]' % (elem+1, len(src_trainer.loss_history)), end='\r')
            self.assertEqual(src_trainer.loss_history[elem], dst_trainer.loss_history[elem])

        print('\n OK')


        print('======== TestDAETrainer.test_save_load <END>')



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
    parser.add_argument('--data-dir',
                        type=str,
                        default='./data/',
                        help='Path to directory to store data (default: data)'
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
