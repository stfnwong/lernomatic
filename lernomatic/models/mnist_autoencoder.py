"""
MNIST_AUTOENCODER
Autoencoder from MNIST dataset

Stefan Wong 2019
"""

import torch.nn as nn

# ---- Autoencoder object ---- #
class MNISTAutoencoder(nn.Module):
    def __init__(self):
        super(MNISTAutoencoder, self).__init__()
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

    def forward(self, X):
        X = self.encoder(X)
        X = self.decoder(X)

        return X
