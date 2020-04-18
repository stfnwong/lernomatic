"""
TEST_DAE_TRAINER
Unit tests for Denoising Autoencoder Trainer

Stefan Wong 2019
"""

import torch
import torchvision
import time
from datetime import timedelta

from lernomatic.models.autoencoder import denoise_ae
from lernomatic.train.autoencoder import dae_trainer
from test import util


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


class TestDAETrainer:
    verbose = True
    test_num_epochs = 4
    data_dir = './data'

    def test_save_load(self) -> None:
        train_dataset, val_dataset = get_mnist_datasets(self.data_dir)

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
            device_id     = util.get_device_id(),
            # trainer params
            batch_size = self.batch_size,
            num_epochs = self.test_num_epochs,
            # disable saving
            save_every = 0,
            print_every = self.print_every,
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
        dst_trainer = dae_trainer.DAETrainer(device_id = util.get_device_id())
        dst_trainer.load_checkpoint(test_checkpoint_file)

        # check the basic trainer params
        assert  src_trainer.num_epochs == dst_trainer.num_epochs
        assert src_trainer.learning_rate == dst_trainer.learning_rate
        assert src_trainer.momentum == dst_trainer.momentum
        assert src_trainer.weight_decay == dst_trainer.weight_decay
        assert src_trainer.loss_function == dst_trainer.loss_function
        assert src_trainer.optim_function == dst_trainer.optim_function
        assert src_trainer.cur_epoch == dst_trainer.cur_epoch
        assert src_trainer.iter_per_epoch == dst_trainer.iter_per_epoch
        assert src_trainer.save_every == dst_trainer.save_every
        assert src_trainer.print_every == dst_trainer.print_every
        assert src_trainer.batch_size == dst_trainer.batch_size
        assert src_trainer.val_batch_size == dst_trainer.val_batch_size
        assert src_trainer.shuffle == dst_trainer.shuffle

        # Now check the models
        src_models = [src_trainer.encoder, src_trainer.decoder]
        dst_models = [dst_trainer.encoder, dst_trainer.decoder]

        for src_mod, dst_mod in zip(src_models, dst_models):

            print('\t Comparing parameters for %s model' % repr(src_mod))
            src_model_params = src_mod.get_net_state_dict()
            dst_model_params = dst_mod.get_net_state_dict()

            assert len(src_model_params.items()) == len(dst_model_params.items())

            # p1, p2 are k,v tuple pairs of each model parameters
            # k = str
            # v = torch.Tensor
            for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
                assert  p1[0] == p2[0]
                print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())), end='\r')
                assert torch.equal(p1[1], p2[1])) is True
            print('\n ...done')

        print('Checking history...')
        dst_trainer.load_history(test_history_file)

        assert len(src_trainer.loss_history) == len(dst_trainer.loss_history)
        for elem in range(len(src_trainer.loss_history)):
            print('Checking loss history element [%d / %d]' % (elem+1, len(src_trainer.loss_history)), end='\r')
            assert src_trainer.loss_history[elem] == dst_trainer.loss_history[elem]

        print('\n OK')
