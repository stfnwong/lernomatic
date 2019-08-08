"""
TEST_PIX2PIX_TRAINER
Unit tests for Pix2PixTrainer module

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
# I suppose that this unit test can almost serve as a model test as well
from lernomatic.models.gan.cycle_gan import resnet_gen
from lernomatic.models.gan.cycle_gan import pixel_disc
from lernomatic.train.gan import pix2pix_trainer
from lernomatic.data.gan import aligned_dataset
from lernomatic.data.gan import gan_transforms

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def get_aligned_dataset(
    ab_path:str,
    dataset_name:str,
    data_root:str,
    transforms=None) -> aligned_dataset.AlignedDatasetHDF5:

    if transforms is None:
        transforms = gan_transforms.get_gan_transforms(
            do_crop = True,
            to_tensor = True,
            do_scale_width = True
        )
    dataset = aligned_dataset.AlignedDataset(
        ab_path,
        data_root = data_root,
        transform = transforms
    )

    return dataset


class TestPix2PixTrainer(unittest.TestCase):
    def setUp(self):
        # TODO : settable?
        self.train_data_root = '/mnt/ml-data/datasets/cyclegan/night2day/train/'
        self.val_data_root   = '/mnt/ml-data/datasets/cyclegan/night2day/val/'
        self.test_num_epochs = 1

    def test_save_load(self):
        print('======== TestPix2PixTrainer.test_save_load_checkpoint ')

        # Get some data
        train_ab_paths = [path for path in os.listdir(self.train_data_root)]
        val_ab_paths   = [path for path in os.listdir(self.val_data_root)]
        train_dataset = get_aligned_dataset(
            train_ab_paths,
            'pix2pix_trainer_test_train_data',
            self.train_data_root
        )
        val_dataset = get_aligned_dataset(
            val_ab_paths,
            'pix2pix_trainer_test_train_data',
            self.val_data_root
        )

        # Get some models - we use resnet and PatchGAN here for now. At some
        # point the bugs in the UnetGenerator also need to be solved and this
        # test should be smaller than a 'real' training run.
        generator     = resnet_gen.ResnetGenerator(3, 3, num_filters=64)
        discriminator = pixel_disc.PixelDiscriminator(3 + 3, num_filters=64)

        test_checkpoint_file = 'checkpoint/pix2pix_trainer_checkpoint_test.pkl'
        test_history_file = 'checkpoint/pix2pix_trainer_history_test.pkl'
        # Get a trainer
        src_trainer = pix2pix_trainer.Pix2PixTrainer(
            generator,
            discriminator,
            # dataset
            train_dataset = train_dataset,
            val_dataset   = val_dataset,
            # trainer general options
            batch_size    = GLOBAL_OPTS['batch_size'],
            device_id     = GLOBAL_OPTS['device_id'],
            num_epochs    = self.test_num_epochs,
            # checkpoint
            save_every    = 0,
            print_every   = GLOBAL_OPTS['print_every'],
        )
        src_trainer.train()

        print('Saving checkpoint to file [%s]' % str(test_checkpoint_file))
        src_trainer.save_checkpoint(test_checkpoint_file)
        print('Saving history to file [%s]' % str(test_history_file))
        src_trainer.save_history(test_history_file)

        # get a new trainer and load
        dst_trainer = pix2pix_trainer.Pix2PixTrainer(None, None, device_id=GLOBAL_OPTS['device_id'])
        dst_trainer.load_checkpoint(test_checkpoint_file)

        # Check that some models were loaded
        self.assertIsNotNone(dst_trainer.g_net)
        self.assertIsNotNone(dst_trainer.d_net)
        self.assertEqual(repr(src_trainer.g_net), repr(dst_trainer.g_net))
        self.assertEqual(repr(src_trainer.d_net), repr(dst_trainer.d_net))

        # check model params
        src_models = [src_trainer.g_net, src_trainer.d_net]
        dst_models = [dst_trainer.g_net, dst_trainer.d_net]

        for src_mod, dst_mod in zip(src_models, dst_models):
            print('Checking parameters for model [%s]' % repr(src_mod))
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

        # check history
        dst_trainer.load_history(test_history_file)
        self.assertIsNotNone(dst_trainer.g_loss_history)
        self.assertIsNotNone(dst_trainer.d_loss_history)
        self.assertEqual(len(src_trainer.g_loss_history), len(dst_trainer.g_loss_history))
        self.assertEqual(len(src_trainer.d_loss_history), len(dst_trainer.d_loss_history))

        for loss_elem in range(len(src_trainer.g_loss_history)):
            print('Checking g_loss_history [%d / %d]' % (loss_elem+1, len(src_trainer.g_loss_history)), end='\r')
            self.assertEqual(src_trainer.g_loss_history[loss_elem], dst_trainer.g_loss_history[loss_elem])
        print('\n OK')

        for loss_elem in range(len(src_trainer.d_loss_history)):
            print('Checking d_loss_history [%d / %d]' % (loss_elem+1, len(src_trainer.g_loss_history)), end='\r')
            self.assertEqual(src_trainer.d_loss_history[loss_elem], dst_trainer.d_loss_history[loss_elem])
        print('\n OK')

        # check the various trainer stats
        self.assertEqual(src_trainer.beta1, dst_trainer.beta1)
        self.assertEqual(src_trainer.l1_lambda, dst_trainer.l1_lambda)
        self.assertEqual(src_trainer.gan_mode, dst_trainer.gan_mode)
        self.assertEqual(src_trainer.learning_rate, dst_trainer.learning_rate)
        self.assertEqual(src_trainer.batch_size, dst_trainer.batch_size)
        self.assertEqual(src_trainer.print_every, dst_trainer.print_every)
        self.assertEqual(src_trainer.save_every, dst_trainer.save_every)
        self.assertEqual(src_trainer.cur_epoch, dst_trainer.cur_epoch)
        self.assertEqual(src_trainer.num_epochs, dst_trainer.num_epochs)

        print('======== TestPix2PixTrainer.test_save_load_checkpoint <END>')


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
                        default=25,
                        help='Print output every time this number of iterations has elapsed'
                        )
    # dataset options
    parser.add_argument('--dataset-root',
                        type=str,
                        default='/mnt/ml-data/datasets/celeba/',
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
