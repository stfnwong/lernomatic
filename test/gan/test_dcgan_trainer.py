"""
TEST_DCGAN_TRAINER
Unit tests for DCGATrainer module

Stefan Wong 2019
"""

import os
import sys
import argparse
import unittest
import torch
import torchvision
from torchvision import transforms
# units under test
from lernomatic.models.gan import dcgan
from lernomatic.train.gan import dcgan_trainer
from lernomatic.data import hdf5_dataset


# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def get_dataset(image_size:int = 64):
    #celeba_transform = transforms.Compose([
    #       transforms.Resize(image_size),
    #       transforms.CenterCrop(image_size),
    #       transforms.ToTensor(),
    #       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #])
    #dataset = torchvision.datasets.ImageFolder(
    #    root=GLOBAL_OPTS['dataset_root'],
    #    transform = celeba_transform
    #)

    dataset = hdf5_dataset.HDF5Dataset(
        GLOBAL_OPTS['dataset_root'],
        feature_name = 'images',
        label_name = 'labels',
        #transform = gan_data_transform
    )

    return dataset


class TestDCGANTrainer(unittest.TestCase):
    def setUp(self) -> None:
        self.verbose = GLOBAL_OPTS['verbose']
        self.test_num_epochs = 1
        self.test_learning_rate = 2e-4

    def test_save_load_checkpoint(self) -> None:
        print('======== TestDCGANTrainer.test_save_load_checkpoint ')

        test_checkpoint = 'checkpoint/dcgan_trainer_test.pkl'
        test_history = 'checkpoint/dcgan_trainer_test_history.pkl'

        train_dataset = get_dataset()
        # get models
        discriminator = dcgan.DCGANDiscriminator()
        generator     = dcgan.DCGANGenerator()
        # get a trainer
        src_trainer = dcgan_trainer.DCGANTrainer(
            D = discriminator,
            G = generator,
            # device
            device_id     = GLOBAL_OPTS['device_id'],
            batch_size    = GLOBAL_OPTS['batch_size'],
            # training params
            train_dataset = train_dataset,
            num_epochs    = self.test_num_epochs,
            learning_rate = self.test_learning_rate,
            verbose       = self.verbose,
            save_every    = 0,
        )
        src_trainer.train()
        print('Saving checkpoint data to file [%s]' % str(test_checkpoint))
        src_trainer.save_checkpoint(test_checkpoint)
        src_trainer.save_history(test_history)

        # load into new trainer
        dst_trainer = dcgan_trainer.DCGANTrainer(
            None,
            None,
            train_dataset = train_dataset,
            device_id = GLOBAL_OPTS['device_id']
        )
        dst_trainer.load_checkpoint(test_checkpoint)
        self.assertEqual(src_trainer.num_epochs, dst_trainer.num_epochs)
        self.assertEqual(src_trainer.learning_rate, dst_trainer.learning_rate)
        self.assertEqual(src_trainer.weight_decay, dst_trainer.weight_decay)
        self.assertEqual(src_trainer.print_every, dst_trainer.print_every)
        self.assertEqual(src_trainer.save_every, dst_trainer.save_every)
        self.assertEqual(src_trainer.device_id, dst_trainer.device_id)

        print('\t Comparing generator model parameters ')
        src_g = src_trainer.generator.get_net_state_dict()
        dst_g = dst_trainer.generator.get_net_state_dict()
        self.assertEqual(len(src_g.items()), len(dst_g.items()))

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_g.items(), dst_g.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_g.items())), end='')
            self.assertEqual(True, torch.equal(p1[1], p2[1]))
            print('\t OK')
        print('\n ...done')

        print('\t Comparing discriminator model parameters')
        src_d = src_trainer.discriminator.get_net_state_dict()
        dst_d = dst_trainer.discriminator.get_net_state_dict()
        self.assertEqual(len(src_d.items()), len(dst_d.items()))

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_d.items(), dst_d.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_d.items())), end='')
            self.assertEqual(True, torch.equal(p1[1], p2[1]))
            print('\t OK')
        print('\n ...done')

        # load history and check
        dst_trainer.load_history(test_history)
        self.assertIsNot(None, dst_trainer.d_loss_history)
        self.assertIsNot(None, dst_trainer.g_loss_history)
        self.assertEqual(len(src_trainer.d_loss_history), len(dst_trainer.d_loss_history))
        self.assertEqual(len(src_trainer.g_loss_history), len(dst_trainer.g_loss_history))

        print('Checking D loss history...')
        for n in range(len(src_trainer.d_loss_history)):
            self.assertEqual(src_trainer.d_loss_history[n], dst_trainer.d_loss_history[n])
        print(' OK')

        print('Checking G loss history...')
        for n in range(len(src_trainer.g_loss_history)):
            self.assertEqual(src_trainer.g_loss_history[n], dst_trainer.g_loss_history[n])
        print(' OK')

        # Try training a bit more. Since the values of cur_epoch and num_epochs
        # are the same, there should be no effect at first
        dst_trainer.train()
        self.assertEqual(dst_trainer.cur_epoch, src_trainer.cur_epoch)

        # If we then adjust the number of epochs (to at least cur_epoch+1) then
        # we should see another
        dst_trainer.set_num_epochs(src_trainer.num_epochs+1)
        dst_trainer.train()
        self.assertEqual(src_trainer.num_epochs + 1, dst_trainer.cur_epoch)

        os.remove(test_checkpoint)
        os.remove(test_history)

        print('======== TestDCGANTrainer.test_save_load_checkpoint <END>')



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
                        default=32,
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
                        default=10,
                        help='Print output every time this number of iterations has elapsed'
                        )
    # dataset options
    parser.add_argument('--dataset-root',
                        type=str,
                        #default='/mnt/ml-data/datasets/celeba/',
                        default='hdf5/dcgan_unit_test.h5',
                        help='Path to root of dataset'
                        )
    # checkpoint options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='checkpoint/',
                        help='Directory to save checkpoints to'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='dcgan-trainer-test',
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
            print('\t[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()