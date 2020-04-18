"""
TEST_MNIST_TRAINER
Unit tests for MNIST trainer object

Stefan Wong 2018
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
# unit(s) under test
from lernomatic.train import mnist_trainer
from lernomatic.models import mnist as mnist_net
from test import util



class TestMNISTTrainer:
    verbose         :bool = True #GLOBAL_OPTS['verbose']
    draw_plot       :bool = True #GLOBAL_OPTS['draw_plot']
    test_num_epochs :int  = 3
    test_batch_size :int  = 16

    def test_save_load_checkpoint_train(self) -> None:
        test_dataset_file = 'hdf5/trainer_unit_test.h5'
        test_checkpoint_name = 'checkpoint/save_load_test_checkpoint.pkl'

        # get a model, trainer
        model = mnist_net.MNISTNet()
        src_tr = mnist_trainer.MNISTTrainer(
            model,
            num_epochs = self.test_num_epochs,
            save_every = 0,
            print_every = 250,
            device_id = util.get_device_id(),
            # dataload options
            checkpoint_name = 'save_load_test',
            batch_size = 16,
            num_workers = 1
            #num_workers = GLOBAL_OPTS['num_workers']
        )

        if self.verbose:
            print('Created trainer object')
            print(src_tr)

        # train for one epoch
        src_tr.train()
        src_tr.save_checkpoint(test_checkpoint_name)

        # Make a new trainer and load all parameters into that
        # I guess we need to put some kind of loader and model here...
        new_model = mnist_net.MNISTNet()
        dst_tr = mnist_trainer.MNISTTrainer(
            new_model,
            device_id = util.get_device_id()
        )
        dst_tr.load_checkpoint(test_checkpoint_name)

        # Test object parameters
        assert src_tr.num_epochs == dst_tr.num_epochs
        assert src_tr.learning_rate == dst_tr.learning_rate
        assert src_tr.weight_decay == dst_tr.weight_decay
        assert src_tr.print_every == dst_tr.print_every
        assert src_tr.save_every == dst_tr.save_every
        assert src_tr.device_id == dst_tr.device_id

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        assert len(src_model_params.items()) == len(dst_model_params.items())

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
            assert p1[0] == p2[0]
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())), end='\r')
            assert torch.equal(p1[1], p2[1]) == True
        print('\n ...done')


    def save_load_checkpoint_train_test(self) -> None:
        model = mnist_net.MNISTNet()

        # Get trainer object
        src_tr = mnist_trainer.MNISTTrainer(
            model,
            save_every = 0,
            print_every = 250,
            checkpoint_name = 'save_load_test',
            device_id = util.get_device_id(),
            # loader options,
            num_epochs = self.test_num_epochs,
            batch_size = self.test_batch_size,
            num_workers = 1
            # Need to provide both training and test datasets
        )

        if self.verbose:
            print('Created trainer object')
            print(src_tr)

        # train for one epoch
        src_tr.train()

        # Now try to load a checkpoint and ensure that there is an
        # acc history attribute that is not None
        dst_tr = mnist_trainer.MNISTTrainer(
            model,
            device_id = util.get_device_id(),
        )
        ck_fname = 'checkpoint/save_load_test_epoch-%d.pkl' % (test_num_epochs-1)
        dst_tr.load_checkpoint(ck_fname)

        # Test object parameters
        assert src_tr.num_epochs == dst_tr.num_epochs
        assert src_tr.learning_rate == dst_tr.learning_rate
        assert src_tr.weight_decay == dst_tr.weight_decay
        assert src_tr.print_every == dst_tr.print_every
        assert src_tr.save_every == dst_tr.save_every
        assert src_tr.device_id == dst_tr.device_id

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        assert len(src_model_params.items()) == len(dst_model_params.items())

        print('\t Comparing model parameters ')
        src_model_params = src_tr.get_model_params()
        dst_model_params = dst_tr.get_model_params()
        assert len(src_model_params.items()) == len(dst_model_params.items())

        # p1, p2 are k,v tuple pairs of each model parameters
        # k = str
        # v = torch.Tensor
        for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
            assert p1[0] == p2[0]
            print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n, len(src_model_params.items())), end='\r')
            assert torch.equal(p1[1], p2[1]) == True
        print('\n ...done')


    def test_save_load_history(self) -> None:
        test_history_name = 'checkpoint/test_history.pkl'

        # get a model, trainer
        model = mnist_net.MNISTNet()
        src_tr = mnist_trainer.MNISTTrainer(
            model,
            num_epochs = self.test_num_epochs,
            save_every = 0,
            print_every = 250,
            device_id = util.get_device_id(),
            # dataload options
            checkpoint_name = 'save_load_test',
            batch_size = 16,
            num_workers = 1
        )

        if self.verbose:
            print('Created trainer object')
            print(src_tr)

        # train for one epoch
        src_tr.train()
        src_tr.save_history(test_history_name)

        # Load history into new object
        dst_tr = mnist_trainer.MNISTTrainer(
            model,
            device_id = util.get_device_id(),
        )
        dst_tr.load_history(test_history_name)

        # Test loss history
        print('\t Comparing loss history....')
        assert src_tr.loss_iter == dst_tr.loss_iter
        for n in range(src_tr.loss_iter):
            print('Checking loss element [%d/%d]' % (n, src_tr.loss_iter), end='\r')
            assert src_tr.loss_history[n] == dst_tr.loss_history[n]

        print('\n ...done')

