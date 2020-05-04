"""
TEST_ADVESARIAL_TRAINER
Unit tests for AAETrainer object

Stefan Wong 2019
"""

import torch
import torchvision
# module(s) under test
from lernomatic.models.autoencoder import aae_common
from lernomatic.train.autoencoder import aae_trainer
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


class TestAAETrainer:
    # MNIST sizes - unit testing on MNIST should be relatively fast
    num_classes     = 10
    hidden_size     = 1000
    x_dim           = 784
    z_dim           = 2
    y_dim           = 10
    test_data_dir   = './data'
    test_num_epochs = 4
    test_batch_size = 64
    print_every     = 200
    verbose         = True

    def test_save_load(self) -> None:
        test_checkpoint_file = 'checkpoint/test_aae_trainer_checkpoint.pth'
        test_history_file = 'checkpoint/test_aae_trainer_history.pth'

        # get some models
        q_net = aae_common.AAEQNet(self.x_dim, self.z_dim, self.hidden_size)
        p_net = aae_common.AAEPNet(self.x_dim, self.z_dim, self.hidden_size)
        d_net = aae_common.AAEDNetGauss(self.z_dim, self.hidden_size)

        train_dataset, val_dataset = get_mnist_datasets(self.test_data_dir)

        # get a trainer
        src_trainer = aae_trainer.AAETrainer(
            q_net,
            p_net,
            d_net,
            # datasets
            train_dataset = train_dataset,
            val_dataset   = val_dataset,
            # train options
            num_epochs    = self.test_num_epochs,
            batch_size    = self.test_batch_size,
            # misc
            print_every   = self.print_every,
            save_every    = 0,
            device_id     = util.get_device_id(),
            verbose       = self.verbose
        )
        # generate the source parameters
        src_trainer.train()
        src_trainer.save_checkpoint(test_checkpoint_file)
        src_trainer.save_history(test_history_file)

        # get a new trainer
        dst_trainer = aae_trainer.AAETrainer(device_id = util.get_device_id())
        dst_trainer.load_checkpoint(test_checkpoint_file)

        # Check parameters of each model in turn
        src_models = [src_trainer.q_net, src_trainer.p_net, src_trainer.d_net]
        dst_models = [dst_trainer.q_net, src_trainer.p_net, src_trainer.d_net]

        for src_mod, dst_mod in zip(src_models, dst_models):

            print('\t Comparing parameters for %s model' % repr(src_mod))
            src_model_params = src_mod.get_net_state_dict()
            dst_model_params = dst_mod.get_net_state_dict()

            assert  len(src_model_params.items()) == len(dst_model_params.items())

            # p1, p2 are k,v tuple pairs of each model parameters
            # k = str
            # v = torch.Tensor
            for n, (p1, p2) in enumerate(zip(src_model_params.items(), dst_model_params.items())):
                assert p1[0] == p2[0]
                print('Checking parameter %s [%d/%d] \t\t' % (str(p1[0]), n+1, len(src_model_params.items())), end='\r')
                assert torch.equal(p1[1], p2[1]) is True
            print('\n ...done')

        # History
        dst_trainer.load_history(test_history_file)
        assert dst_trainer.d_loss_history is not None
        assert dst_trainer.g_loss_history is not None
        assert dst_trainer.recon_loss_history is not None

        assert len(src_trainer.d_loss_history) == len(dst_trainer.d_loss_history)
        assert len(src_trainer.g_loss_history) == len(dst_trainer.g_loss_history)
        assert len(src_trainer.recon_loss_history) == len(dst_trainer.recon_loss_history)
