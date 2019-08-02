"""
DCGAN
A more flexible DCGAN implementation

Stefan Wong 2019
"""

import torch
import torch.nn as nn
import numpy as np
from lernomatic.models import common

# debug
from pudb import set_trace; set_trace()



class DCGANGenBlock(nn.Module):
    """
    DCGANGenBlock
    A single block of convolutions for a DCGAN Generator
    """
    def __init__(self, **kwargs) -> None:
        self.num_input_filters:int  = kwargs.pop('num_input_filters', 64)
        self.num_output_filters:int = kwargs.pop('num_output_filters', 64)
        self.kernel_size:int        = kwargs.pop('kernel_size', 4)
        self.stride:int             = kwargs.pop('stride', 2)
        self.padding:int            = kwargs.pop('padding', 0)

        # FIXME: debug (remove)
        print('num_input_filters  : %d' % self.num_input_filters)
        print('num_output_filters : %d' % self.num_output_filters)

        super(DCGANGenBlock, self).__init__()
        # network structure
        self.convtranspose = nn.ConvTranspose2d(
            self.num_input_filters,
            self.num_output_filters,
            kernel_size = self.kernel_size,
            stride      = self.stride,
            padding     = self.padding,
            bias        = False
        )
        self.bn   = nn.BatchNorm2d(self.num_output_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.convtranpose(X)
        out = self.bn(out)
        out = self.relu(out)

        return out



class DCGANGeneratorModule(nn.Module):
    """
    DCGANGeneratorModule
    Holds a collection of DCGANGenBlock modules. The total number of blocks in the module
    is equal to num_blocks+1, as the last block will apply a tanh() function to the outputs
    and scale.

    Arguments:
        num_channels: (int)
            Number of channels in output image (default: 3)

        num_filters: (int)
            Base number filters to use (default: 64)

        img_size: (int)
            Size of output image. Output image is square (default: 64)

        zvec_dim: (int)
            Dimensions in latent Z vector (default: 100)
    """
    def __init__(self, **kwargs) -> None:
        self.num_channels:int = kwargs.pop('num_channels', 3)
        # TODO : I think its better to compute the number of required blocks
        # based on the image size. So 64x64 will require 5 blocks, 128x128 will
        # require 6 blocks, etc
        #self.num_blocks:int   = kwargs.pop('num_blocks', 4)
        self.num_filters:int  = kwargs.pop('num_filters', 64)
        self.kernel_size:int  = kwargs.pop('kernel_size', 4)
        self.img_size:int     = kwargs.pop('img_size', 64)
        self.zvec_dim:int     = kwargs.pop('zvec_dim', 100)

        # enforce that image size must be a power of 2

        super(DCGANGeneratorModule, self).__init__()

        # -2 here since we specify the input vector projection and image output
        # blocks outside of the loop
        self.num_blocks = int(np.ceil(np.log2(self.img_size))) - 2
        fscale = 2 ** (self.num_blocks)

        # TODO : issue here with construction...
        gen_blocks = []
        # Initial projection of Z vector to first conv layer
        gen_blocks += [DCGANGenBlock(
            num_input_filters  = self.zvec_dim,
            num_output_filters = self.num_filters * fscale,
            kernel_size        = self.kernel_size,
            stride             = 1,
            padding            = 0
            )
        ]
        for b in range(self.num_blocks):
            gen_blocks += [
                DCGANGenBlock(
                    num_input_filters  = self.num_filters * fscale,
                    num_output_filters = self.num_filters * (fscale // 2),
                    kernel_size        = self.kernel_size,
                    stride             = 2,
                    padding            = 1
                )
            ]
            fscale = fscale // 2

        self.blocks = nn.Sequential(*gen_blocks)

        # 'Final' block
        self.final_convtranspose = nn.ConvTranspose2d(
            self.num_filters,
            self.num_channels,
            kernel_size = self.kernel_size,
            stride = 2,
            padding = 1,
            bias = False
        )
        self.final_tan = nn.Tanh()

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.blocks(X)
        out = self.final_convtranspose(out)
        out = self.final_tan(out)

        return out


class DCGANGenerator(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = DCGANGeneratorModule(**kwargs)
        self.model_name         = 'DCGANGenerator'
        self.module_name        = 'DCGANGeneratorModule'
        self.import_path        = 'lernomatic.models.gan.dcgan'
        self.module_import_path = 'lernomatic.models.gan.dcgan'

        self.init_weights()

    def __repr__(self) -> str:
        return 'DCGGenerator'

    def get_zvec_dim(self) -> int:
        return self.net.zvec_dim

    def get_num_filters(self) -> int:
        return self.net.num_filters

    def zero_grad(self) -> None:
        self.net.zero_grad()

    def init_weights(self) -> None:
        classname = self.net.__class__.__name__
        if classname.find('Conv') != -1:
            self.net.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            self.net.weight.data.normal_(1.0, 0.02)
            self.net.bias.data.fill_(0)


class DCGANDiscBlock(nn.Module):
    """
    DCGANDiscBlock
    A single block of convolutions for a DCGAN discriminator
    """
    def __init__(self, **kwargs) -> None:
        self.num_input_filters:int  = kwargs.pop('num_input_filters', 64)
        self.num_output_filters:int = kwargs.pop('num_output_filters', 64)
        self.kernel_size:int        = kwargs.pop('kernel_size', 4)
        self.stride:int             = kwargs.pop('stride', 2)
        self.padding:int            = kwargs.pop('padding', 0)
        self.relu_leak:float        = kwargs.pop('relu_leak', 0.2)

        super(DCGANDiscBlock, self).__init__()
        # network structure
        self.conv = nn.Conv2d(
            self.num_input_filters,
            self.num_output_filters,
            kernel_size = self.kernel_size,
            stride      = self.stride,
            padding     = self.padding,
            bias        = False
        )
        self.bn = nn.BatchNorm2d(self.num_output_filters)
        self.lrelu = nn.LeakyReLU(self.relu_leak, inplace=True)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.conv(X)
        out = self.bn(out)
        out = self.lrelu(out)

        return out



class DCGANDiscriminatorModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.num_filters:int  = kwargs.pop('num_filters', 64)
        self.num_channels:int = kwargs.pop('num_channels', 3)
        self.kernel_size:int  = kwargs.pop('kernel_size', 4)
        self.img_size:int     = kwargs.pop('img_size', 64)

        super(DCGANDiscriminatorModule, self).__init__()

        self.num_blocks = int(np.ceil(np.log2(self.img_size)))-1

        disc_blocks = []
        fscale = 1
        for b in range(self.num_blocks):
            if b == 0:
                disc_blocks += [
                    DCGANDiscBlock(
                        num_input_filters  = self.num_channels,
                        num_output_filters = self.num_filters,
                        kernel_size = self.kernel_size,
                        stride = 2,
                        padding = 1
                    )
                ]
            else:
                disc_blocks += [
                    DCGANDiscBlock(
                        num_input_filters = self.num_filters * fscale,
                        num_output_filters = self.num_filters * (fscale * 2),
                        kernel_size = self.kernel_size,
                        stride = 2,
                        padding = 1
                    )
                ]
                fscale = fscale * 2

        self.blocks = nn.Sequential(*disc_blocks)
        self.final_conv = nn.Conv2d(
            self.num_filters * fscale,
            1,
            kernel_size = self.kernel_size,
            stride = 1,
            padding = 0,
            bias = False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.blocks(X)
        out = self.final_conv(out)
        out = self.sigmoid(out)

        return out



class DCGANDiscriminator(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = DCGANDiscriminatorModule(**kwargs)
        # internal bookkeeping
        self.model_name         = 'DCGANDiscriminator'
        self.module_name        = 'DCGANDiscriminatorModule'
        self.import_path        = 'lernomatic.models.gan.dcagn'
        self.module_import_path = 'lernomatic.models.gan.dcagn'

        self.init_weights()

    def __repr__(self) -> str:
        return 'DCGDiscriminator'

    def get_num_filters(self) -> int:
        return self.net.num_filters

    def zero_grad(self) -> None:
        self.net.zero_grad()

    def init_weights(self) -> None:
        classname = self.net.__class__.__name__
        if classname.find('Conv') != -1:
            self.net.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            self.net.weight.data.normal_(1.0, 0.02)
            self.net.bias.data.fill_(0)
