"""
MNIST_AUTOENCODER
Autoencoder from MNIST dataset

Stefan Wong 2019
"""

import torch
import torch.nn as nn
from lernomatic.models import common


class MNISTAutoencoder(common.LernomaticModel):
    def __init__(self) -> None:
        self.net = MNISTAutoencoderModule()
        self.model_name = 'MNISTAutoencoder'
        self.module_name = 'MNISTAutoencoderModule'
        self.import_path = 'lernomatic.models.mnist_autoencoder'
        self.module_import_path = 'lernomatic.models.mnist_autoencoder'

    def __repr__(self) -> str:
        return 'MNISTAutoencoder'


# ---- Autoencoder object ---- #
class MNISTAutoencoderModule(nn.Module):
    def __init__(self) -> None:
        super(MNISTAutoencoderModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, X) -> torch.Tensor:
        X = self.encoder(X)
        X = self.decoder(X)

        return X
