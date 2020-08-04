"""
TEST_PIX2PIX_TRAINER
Unit tests for Pix2PixTrainer module

Stefan Wong 2019
"""

import os
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
from test import util


def get_aligned_dataset(
    ab_path:str,
    dataset_name:str,
    data_root:str,
    transforms=None) -> aligned_dataset.AlignedDataset:

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


class TestPix2PixTrainer:
    # TODO : settable?
    train_data_root = '/mnt/ml-data/datasets/cyclegan/night2day/train/'
    val_data_root   = '/mnt/ml-data/datasets/cyclegan/night2day/val/'
    test_dataset    = 'hdf5/night2day-unittest-256.h5'
    test_num_epochs = 1
    batch_size      = 16
    print_every     = 250

    @pytest.mark.skip(reason='Need a way to test this without relying on my data path')
    def test_save_load(self) -> None:
        # Get some data
        train_ab_paths = [path for path in os.listdir(self.train_data_root)]
        val_ab_paths   = [path for path in os.listdir(self.val_data_root)]

        train_dataset = aligned_dataset.AlignedDatasetHDF5(
            self.test_dataset
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
            val_dataset   = None,
            # trainer general options
            batch_size    = self.batch_size,
            device_id     = util.get_device_id(),
            num_epochs    = self.test_num_epochs,
            # checkpoint
            save_every    = 0,
            print_every   = self.print_every,
        )
        src_trainer.train()

        print('Saving checkpoint to file [%s]' % str(test_checkpoint_file))
        src_trainer.save_checkpoint(test_checkpoint_file)
        print('Saving history to file [%s]' % str(test_history_file))
        src_trainer.save_history(test_history_file)

        # get a new trainer and load
        dst_trainer = pix2pix_trainer.Pix2PixTrainer(None, None, device_id=util.get_device_id())
        dst_trainer.load_checkpoint(test_checkpoint_file)

        # Check that some models were loaded
        assert dst_trainer.g_net is not None
        assert dst_trainer.d_net is not None
        assert repr(src_trainer.g_net) == repr(dst_trainer.g_net)
        assert repr(src_trainer.d_net) == repr(dst_trainer.d_net)

        # check model params
        src_models = [src_trainer.g_net, src_trainer.d_net]
        dst_models = [dst_trainer.g_net, dst_trainer.d_net]

        for src_mod, dst_mod in zip(src_models, dst_models):
            print('Checking parameters for model [%s]' % repr(src_mod))
            src_model_params = src_mod.get_net_state_dict()
            dst_model_params = dst_mod.get_net_state_dict()

            assert len(src_model_params.items()) == len(dst_model_params.items())

            # p1, p2 are k,v tuple pairs of each model parameters
            # k = str
            # v = torch.Tensor
            for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
                assert p1[0] == p2[0]
                print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())), end='\r')
                assert torch.equal(p1[1], p2[1]) is True
            print('\n ...done')

        # check the various trainer stats
        assert src_trainer.beta1 == dst_trainer.beta1
        assert src_trainer.l1_lambda == dst_trainer.l1_lambda
        assert src_trainer.gan_mode == dst_trainer.gan_mode
        assert src_trainer.learning_rate == dst_trainer.learning_rate
        assert src_trainer.batch_size == dst_trainer.batch_size
        assert src_trainer.print_every == dst_trainer.print_every
        assert src_trainer.save_every == dst_trainer.save_every
        assert src_trainer.cur_epoch == dst_trainer.cur_epoch
        assert src_trainer.num_epochs == dst_trainer.num_epochs

        # check history
        dst_trainer.load_history(test_history_file)
        assert dst_trainer.g_loss_history is not None
        assert dst_trainer.d_loss_history is not None
        assert len(src_trainer.g_loss_history) == len(dst_trainer.g_loss_history)
        assert len(src_trainer.d_loss_history) == len(dst_trainer.d_loss_history)

        for loss_elem in range(len(src_trainer.g_loss_history)):
            print('Checking g_loss_history [%d / %d]' % (loss_elem+1, len(src_trainer.g_loss_history)), end='\r')
            assert src_trainer.g_loss_history[loss_elem] == dst_trainer.g_loss_history[loss_elem]
        print('\n OK')

        for loss_elem in range(len(src_trainer.d_loss_history)):
            print('Checking d_loss_history [%d / %d]' % (loss_elem+1, len(src_trainer.g_loss_history)), end='\r')
            assert  src_trainer.d_loss_history[loss_elem] == dst_trainer.d_loss_history[loss_elem]
        print('\n OK')

        # we need to set the train loaders
        dst_trainer.set_train_dataset(train_dataset)
        # Now try to extent the trainer history and train for another epoch
        dst_trainer.set_num_epochs(src_trainer.num_epochs+1)
        assert  dst_trainer.cur_epoch == src_trainer.cur_epoch

        print('Continuing training for dst_trainer from epoch %d' % dst_trainer.cur_epoch)
        dst_trainer.train()
        assert src_trainer.num_epochs+1 == dst_trainer.cur_epoch
