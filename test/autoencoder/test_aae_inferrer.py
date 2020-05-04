"""
TEST_AAE_INFERRER

Stefan Wong 2019
"""

import torch
import torchvision
# module(s) under test
from lernomatic.models.autoencoder import aae_common
from lernomatic.train.autoencoder import aae_trainer
from lernomatic.infer.autoencoder import aae_inferrer


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


def get_device_id() -> int:
    if torch.cuda.is_available():
        return 0
    return -1


class TestAAEInferrer:
    num_classes      = 10
    hidden_size      = 1000
    x_dim            = 784
    z_dim            = 2
    y_dim            = 10
    test_data_dir    = './data'
    test_num_epochs  = 4
    test_batch_size  = 32
    test_print_every = 25
    verbose          = True

    def test_infer(self) -> None:
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
            batch_size    = self.test_batch_size,
            # misc
            print_every   = self.test_print_every,
            save_every    = 0,
            device_id     = get_device_id(),
            verbose       = self.verbose
        )
        # train
        trainer.train()

        # get an inferrer
        inferrer = aae_inferrer.AAEInferrer(
            q_net,
            p_net,
            device_id = get_device_id()
        )

        # perform inference on trained models
        for batch_idx, (data, target) in enumerate(trainer.val_loader):
            data.resize_(self.test_batch_size, inferrer.q_net.get_x_dim())
            gen_img = inferrer.forward(data)
            assert gen_img is not None
            #img_filename = 'figures/aae/aae_batch_%d.png' % int(batch_idx)
            #recon_to_plt(ax, data.cpu(), gen_img.cpu())
            #fig.savefig(img_filename, bbox_inches='tight')

