"""
TEST_AAE_SEMI_TRAINER
Unit tests for Semi-supervised Adversarial Autoencoder Trainer

Stefan Wong 2019
"""

import torch
import torchvision
# module(s) under test
from lernomatic.models.autoencoder import aae_common
from lernomatic.train.autoencoder import aae_semisupervised_trainer
from lernomatic.data.mnist import mnist_sub
from test import util


class TestAAESemiTrainer:
    # MNIST sizes - unit testing on MNIST should be relatively fast
    num_classes     = 10
    hidden_size     = 1000
    x_dim           = 784
    z_dim           = 2
    y_dim           = 10
    test_data_dir   = './data'
    test_num_epochs = 4
    test_batch_size = 32
    print_every     = 25
    transform       = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize( (0.1307,), (0.3081,))
    ])
    verbose         = True
    print_every     = 200
    batch_size      = 32

    def test_save_load(self) -> None:
        test_checkpoint_file = 'checkpoint/test_aae_semi_trainer_checkpoint.pth'
        test_history_file    = 'checkpoint/test_aae_semi_trainer_history.pth'

        q_net = aae_common.AAEQNet(
            self.x_dim,
            self.z_dim,
            self.hidden_size,
            num_classes = self.num_classes
        )
        p_net = aae_common.AAEPNet(
            self.x_dim,
            self.z_dim+ self.num_classes,
            self.hidden_size
        )
        d_cat_net = aae_common.AAEDNetGauss(
            self.num_classes,
            self.hidden_size
        )
        d_gauss_net = aae_common.AAEDNetGauss(
            self.z_dim,
            self.hidden_size
        )

        q_net.set_cat_mode()
        assert q_net.net.cat_mode == True

        # We also need to sub-sample some parts of the MNIST dataset to produce the
        # 'labelled' data loaders
        print('Creating MNIST sub-dataset...')
        train_label_dataset, val_label_dataset, train_unlabel_dataset = mnist_sub.gen_mnist_subset(
            self.test_data_dir,
            transform = self.transform,
            verbose = self.verbose
        )

        assert train_label_dataset is not None
        assert train_unlabel_dataset is not None
        assert val_label_dataset is not None

        src_trainer = aae_semisupervised_trainer.AAESemiTrainer(
            q_net,
            p_net,
            d_cat_net,
            d_gauss_net,
            # datasets
            train_label_dataset   = train_label_dataset,
            train_unlabel_dataset = train_unlabel_dataset,
            val_label_dataset     = val_label_dataset,
            # train options
            num_epochs    = self.test_num_epochs,
            batch_size    = self.batch_size,
            # misc
            print_every   = self.print_every,
            save_every    = 0,
            device_id     = util.get_device_id(),
            verbose       = self.verbose
        )

        src_trainer.train()
        print('Saving checkpoint to file [%s]' % str(test_checkpoint_file))
        src_trainer.save_checkpoint(test_checkpoint_file)

        dst_trainer = aae_semisupervised_trainer.AAESemiTrainer(device_id = util.get_device_id())
        assert dst_trainer.q_net is None
        assert dst_trainer.p_net is None
        assert dst_trainer.d_cat_net is None
        assert dst_trainer.d_gauss_net is None

        # Test that models, etc are loaded
        print('Loading checkpoint data from [%s]' % str(test_checkpoint_file))
        dst_trainer.load_checkpoint(test_checkpoint_file)
        assert dst_trainer.q_net is not None
        assert dst_trainer.p_net is not None
        assert dst_trainer.d_cat_net is not None
        assert dst_trainer.d_gauss_net is not None

        model_list = ['q_net', 'p_net', 'd_cat_net', 'd_gauss_net']

        for model in model_list:
            src_model = getattr(src_trainer, model)
            dst_model = getattr(dst_trainer, model)
            assert src_model is not None
            assert dst_model is not None
            print('\t Comparing parameters for model [%s]' % repr(src_model))
            src_model_params = src_model.get_net_state_dict()
            dst_model_params = dst_model.get_net_state_dict()

            assert len(src_model_params.items()) == len(dst_model_params.items())

            # p1, p2 are k,v tuple pairs of each model parameters
            # k = str
            # v = torch.Tensor
            for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
                assert p1[0] == p2[0]
                print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())), end='\r')
                assert torch.equal(p1[1], p2[1]) is True
            print('\n ...done')


        # Test that history is correctly loaded
        print('Saving history to file [%s]' % str(test_history_file))
        src_trainer.save_history(test_history_file)
        dst_trainer.load_history(test_history_file)

        # Check iteration values
        assert src_trainer.loss_iter == dst_trainer.loss_iter
        assert src_trainer.val_loss_iter == dst_trainer.val_loss_iter
        assert src_trainer.train_val_loss_iter == dst_trainer.train_val_loss_iter
        assert src_trainer.acc_iter == dst_trainer.acc_iter
        assert src_trainer.cur_epoch == dst_trainer.cur_epoch
        assert src_trainer.iter_per_epoch == dst_trainer.iter_per_epoch

        # Check history arrays
        assert dst_trainer.d_loss_history is not None
        assert dst_trainer.g_loss_history is not None
        assert dst_trainer.recon_loss_history is not None
        assert dst_trainer.class_loss_history is not None

        assert len(src_trainer.d_loss_history) == len(dst_trainer.d_loss_history)
        assert len(src_trainer.g_loss_history) == len(dst_trainer.g_loss_history)
        assert len(src_trainer.recon_loss_history) == len(dst_trainer.recon_loss_history)
        assert len(src_trainer.class_loss_history) == len(dst_trainer.class_loss_history)

        for idx in range(len(src_trainer.d_loss_history)):
            print('Checking d_loss_history idx [%d / %d]' % (idx+1, len(src_trainer.d_loss_history)), end='\r')
            assert src_trainer.d_loss_history[idx] == dst_trainer.d_loss_history[idx]
        print('\n OK')

        for idx in range(len(src_trainer.g_loss_history)):
            print('Checking d_loss_history idx [%d / %d]' % (idx+1, len(src_trainer.g_loss_history)), end='\r')
            assert src_trainer.g_loss_history[idx] == dst_trainer.g_loss_history[idx]
        print('\n OK')

        for idx in range(len(src_trainer.recon_loss_history)):
            print('Checking d_loss_history idx [%d / %d]' % (idx+1, len(src_trainer.recon_loss_history)), end='\r')
            assert src_trainer.recon_loss_history[idx] == dst_trainer.recon_loss_history[idx]
        print('\n OK')

        for idx in range(len(src_trainer.class_loss_history)):
            print('Checking d_loss_history idx [%d / %d]' % (idx+1, len(src_trainer.class_loss_history)), end='\r')
            assert src_trainer.class_loss_history[idx] == dst_trainer.class_loss_history[idx]
        print('\n OK')
