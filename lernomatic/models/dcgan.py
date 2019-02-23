"""
DCGAN
Modules for DCGAN

"""

import torch.nn as nn

# Generator implementation
class DCGGenerator(nn.Module):
    def __init__(self, **kwargs):
        super(DCGGenerator, self).__init__()
        self.zvec_dim = kwargs.pop('zvec_dim', 100)
        self.num_filters = kwargs.pop('num_filters', 64)
        self.num_channels = kwargs.pop('num_channels', 3)     # number of channels in output image
        self.main = nn.Sequential(
            # z is input going into covolution
            nn.ConvTranspose2d(self.zvec_dim,
                               self.num_filters * 8,
                               4, 1, 0,
                               bias=False),
            nn.BatchNorm2d(self.num_filters * 8),
            nn.ReLU(True),
            # state size, (nggfg * 8) x 4 x 4
            nn.ConvTranspose2d(self.num_filters * 8,
                               self.num_filters * 4,
                               4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(self.num_filters * 4),
            nn.ReLU(True),
            # state size (num_filters * 4) x 8 x 8
            nn.ConvTranspose2d(self.num_filters * 4,
                               self.num_filters * 2,
                               4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(self.num_filters * 2),
            nn.ReLU(True),
            # state size (num_filters*2) x 16 x 16
            nn.ConvTranspose2d(self.num_filters * 2,
                               self.num_filters,
                               4, 2, 1,
                               bias=False),
            nn.BatchNorm2d(self.num_filters),
            nn.ReLU(True),
            # state size (num_filters) x 32 x 32
            nn.ConvTranspose2d(self.num_filters,
                               self.num_channels,
                               4, 2, 1,
                               bias=False),
            nn.Tanh()
            # state size (num_channels) x 64 x 64
        )

    def forward(self, X):
        return self.main(X)

    def get_ln_id(self):
        return 'DCGGenerator-%d (%d filters)' % (self.zvec_dim, self.num_filters)

    def get_zvec_dim(self):
        return self.zvec_dim


class DCGDiscriminator(nn.Module):
    def __init__(self, **kwargs):
        super(DCGDiscriminator, self).__init__()
        self.num_filters  = kwargs.pop('num_filters', 64)
        self.num_channels = kwargs.pop('num_channels', 3)
        self.main = nn.Sequential(
            # input is (num_channels) x 64 x 64
            nn.Conv2d(self.num_channels,
                      self.num_filters,
                      4, 2, 1,
                      bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (num_filters) x 32 x 32
            nn.Conv2d(self.num_filters,
                      self.num_filters * 2,
                      4, 2, 1,
                      bias=False),
            nn.BatchNorm2d(self.num_filters * 2),
            nn.LeakyReLU(0.2,
                         inplace=True),
            # state size (num_filters * 2) x 16 x 16
            nn.Conv2d(self.num_filters * 2,
                      self.num_filters * 4,
                      4, 2, 1,
                      bias=False),
            nn.BatchNorm2d(self.num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (num_filters * 2) x 8 x 8
            nn.Conv2d(self.num_filters * 4,
                      self.num_filters * 8,
                      4, 2, 1,
                      bias=False),
            nn.BatchNorm2d(self.num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (num_filters * 8) x 4 x 4
            nn.Conv2d(self.num_filters * 8,
                      1,
                      4, 1, 0,
                      bias=False),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.main(X)

    def get_ln_id(self):
        return 'DCGDiscriminator (%d filters)' % self.num_filters
