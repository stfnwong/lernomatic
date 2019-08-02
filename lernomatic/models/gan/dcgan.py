"""
DCGAN
A more flexible DCGAN implementation

Stefan Wong 2019
"""

import torch
import torch.nn as nn
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

    """
    def __init__(self, **kwargs) -> None:
        self.num_channels:int = kwargs.pop('num_channels', 3)
        self.num_blocks:int   = kwargs.pop('num_blocks', 4)
        self.num_filters:int  = kwargs.pop('num_filters', 64)
        self.kernel_size:int  = kwargs.pop('kernel_size', 4)
        self.zvec_dim:int     = kwargs.pop('zvec_dim', 100)

        super(DCGANGeneratorModule, self).__init__()

        gen_blocks = []
        for b in range(self.num_blocks):
            gen_blocks += [
                DCGANGenBlock(
                )
            ]



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
        pass



class DCGANGenerator(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = DCGGeneratorModule(**kwargs)
        self.model_name         = 'DCGGenerator'
        self.module_name        = 'DCGGeneratorModule'
        self.import_path        = 'lernomatic.models.gan.dcgan'
        self.module_import_path = 'lernomatic.models.gan.dcgan'

    def __repr__(self) -> str:
        return 'DCGGenerator'

    def get_zvec_dim(self) -> int:
        return self.net.zvec_dim

    def get_num_filters(self) -> int:
        return self.net.num_filters

    def zero_grad(self) -> None:
        self.net.zero_grad()



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
            self.num_input_filters
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

        super(DCGANDiscriminatorModule, self).__init__()
