"""
LRGANGAN
Layered-Recursive GAN.
Models that implement the Layered Recursive GAN (https://arxiv.org/pdf/1703.01560.pdf)

Stefan Wong 2019
"""

import torch
import torch.nn as nn
import numpy as np

from lernomatic.models import common
from lernomatic.util import math_util

# TODO: to save some time just use the DCGAN blocks here.



# ======== GENERATOR ======== #
class LRGANGenerator(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = LRGANGeneratorModule(**kwargs)
        self.import_path       : str = 'lernomatic.model.gan.lrgan'
        self.model_name        : str = 'LRGANGenerator'
        self.module_name       : str = 'LRGANGeneratorModule'
        self.module_import_path: str = 'lernomatic.model.gan.lrgan'

    def __repr__(self) -> str:
        return 'LRGANGenerator'



class LRGANGeneratorModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.nz: int = kwargs.pop('nz', 128)
        self.ngf: int = kwargs.pop('ngf', 64)       # num generator features?

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        pass







# ======== DISCRIMINATOR ======== #
class LRGANDiscriminator(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = LRGANDiscriminatorModule(**kwargs)
        self.import_path       : str = 'lernomatic.model.gan.lrgan'
        self.model_name        : str = 'LRGANDiscriminator'
        self.module_name       : str = 'LRGANDiscriminatorModule'
        self.module_import_path: str = 'lernomatic.model.gan.lrgan'

    def __repr__(self) -> str:
        return 'LRGANDiscriminator'

    def get_num_blocks(self) -> int:
        return self.net.num_blocks




class LRGANDiscriminatorModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.num_filters:int  = kwargs.pop('num_filters', 64)
        self.num_channels:int = kwargs.pop('num_channels', 3)
        self.kernel_size:int  = kwargs.pop('kernel_size', 4)
        self.img_size:int     = kwargs.pop('img_size', 64)

        super(LRGANDiscriminatorModule, self).__init__()

        if not math_util.is_pow2_int(self.img_size):
            raise ValueError('Image size [%d] must be a power of 2' % str(self.img_size))

        # construct the network
        disc_blocks = []
        self.num_blocks = int(np.ceil(np.log2(self.img_size)))-2

        conv_depth_in = self.num_channels
        conv_depth_out = self.num_filters
        for b in range(self.num_blocks):
            # debug
            print('Layer %d : conv_depth_in : %d, conv_depth_out: %d' %\
                  (b, conv_depth_in, conv_depth_out)
            )
            disc_blocks += [
                nn.Conv2d(
                    conv_depth_in,
                    conv_depth_out,
                    kernel_size = self.kernel_size,
                    stride = 2,
                    padding = 1,
                    bias = False
                )
            ]
            if b > 0:
                disc_blocks += [
                    nn.BatchNorm2d(conv_depth_out)
                ]
            disc_blocks += [
                nn.LeakyReLU(0.2, inplace=True)
            ]

            conv_depth_in = conv_depth_out
            conv_depth_out = 2 * conv_depth_out

        self.net = nn.Sequential(*disc_blocks)
        self.final_conv = nn.Conv2d(
            conv_depth_in,
            1,
            kernel_size = self.kernel_size,
            stride = 1,
            padding = 0,
            bias = False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.net(X)
        out = self.final_conv(out)
        out = self.sigmoid(out)

        return out.view(-1, 1)

