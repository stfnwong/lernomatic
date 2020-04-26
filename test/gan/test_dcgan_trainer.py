"""
TEST_DCGAN_TRAINER
Unit tests for DCGATrainer module

Stefan Wong 2019
"""

import os
import torch
import torchvision
from torchvision import transforms
# units under test
from lernomatic.models.gan import dcgan
from lernomatic.train.gan import dcgan_trainer
from lernomatic.data import hdf5_dataset
from test import util

DATASET_ROOT = 'hdf5/dcgan_unit_test.h5'

def get_dataset(image_size:int = 64):
    dataset = hdf5_dataset.HDF5Dataset(
        DATASET_ROOT,
        feature_name = 'images',
        label_name = 'labels',
        #transform = gan_data_transform
    )

    return dataset


class TestDCGANTrainer:
    verbose = True
    test_num_epochs    = 1
    test_learning_rate = 2e-4
    batch_size         = 16
    print_every        = 250

    def test_save_load_checkpoint(self) -> None:
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
            device_id     = util.get_device_id(),
            batch_size    = self.batch_size,
            # training params
            train_dataset = train_dataset,
            num_epochs    = self.test_num_epochs,
            learning_rate = self.test_learning_rate,
            verbose       = self.verbose,
            print_every   = self.print_every,
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
            device_id = util.get_device_id()
        )
        dst_trainer.load_checkpoint(test_checkpoint)
        assert  src_trainer.num_epochs == dst_trainer.num_epochs
        assert  src_trainer.learning_rate == dst_trainer.learning_rate
        assert  src_trainer.weight_decay == dst_trainer.weight_decay
        assert  src_trainer.print_every == dst_trainer.print_every
        assert  src_trainer.save_every == dst_trainer.save_every
        assert  src_trainer.device_id == dst_trainer.device_id

        print('\t Comparing generator model parameters ')
        src_g = src_trainer.generator.get_net_state_dict()
        dst_g = dst_trainer.generator.get_net_state_dict()
        assert len(src_g.items()) == len(dst_g.items())

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_g.items(), dst_g.items())):
            assert p1[0] == p2[0]
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_g.items())), end='')
            assert torch.equal(p1[1], p2[1]) is True
            print('\t OK')
        print('\n ...done')

        print('\t Comparing discriminator model parameters')
        src_d = src_trainer.discriminator.get_net_state_dict()
        dst_d = dst_trainer.discriminator.get_net_state_dict()
        assert len(src_d.items()) == len(dst_d.items())

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_d.items(), dst_d.items())):
            assert  p1[0] == p2[0]
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_d.items())), end='')
            assert torch.equal(p1[1], p2[1]) is True
            print('\t OK')
        print('\n ...done')

        # load history and check
        dst_trainer.load_history(test_history)
        assert dst_trainer.d_loss_history is not None
        assert dst_trainer.g_loss_history is not None
        assert len(src_trainer.d_loss_history) == len(dst_trainer.d_loss_history)
        assert len(src_trainer.g_loss_history) == len(dst_trainer.g_loss_history)

        print('Checking D loss history...')
        for n in range(len(src_trainer.d_loss_history)):
            assert  src_trainer.d_loss_history[n] == dst_trainer.d_loss_history[n]
        print(' OK')

        print('Checking G loss history...')
        for n in range(len(src_trainer.g_loss_history)):
            assert  src_trainer.g_loss_history[n] == dst_trainer.g_loss_history[n]
        print(' OK')

        # Try training a bit more. Since the values of cur_epoch and num_epochs
        # are the same, there should be no effect at first
        dst_trainer.train()
        assert dst_trainer.cur_epoch == src_trainer.cur_epoch

        # If we then adjust the number of epochs (to at least cur_epoch+1) then
        # we should see another
        dst_trainer.set_num_epochs(src_trainer.num_epochs+1)
        dst_trainer.train()
        assert  src_trainer.num_epochs + 1 == dst_trainer.cur_epoch

        os.remove(test_checkpoint)
        os.remove(test_history)
