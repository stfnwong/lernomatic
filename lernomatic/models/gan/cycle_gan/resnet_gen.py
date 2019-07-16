"""
RESNET_GENERATOR
A Resnet-based generator module made from a few resnet blocks
and some downsampling operations.

Adapted in large part from models in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Stefan Wong 2019
"""

import functools
import torch
import torch.nn as nn
from lernomatic.models import common
from lernomatic.models.cycle_gan import gan_resnet


def get_norm_layer(norm_type:str='instance') -> nn.Module:

    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2d,
            affine=True,
            track_running_stats=True
        )
    elif norm_layer == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False
        )
    elif norm_layer == 'none':
        norm_layer = lambda x: IdentityModule()
    else:
        raise NotImplementedError('[%s] normalization layer not supported' % str(norm_type))

    return norm_layer



class IdentityModule(nn.Module):
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return X


class ResnetGenerator(common.LernomaticModel):
    def __init__(self,
                 input_channels:int=3,
                 output_channels:int=3,
                 **kwargs) -> None:
        self.net = ResnetGeneratorModule(
            input_channels,
            output_channels,
            **kwargs)
        self.import_path = 'lernomatic.models.cycle_gan.resnet_gen'
        self.module_import_path = 'lernomatic.models.cycle_gan.resnet_gen'
        self.model_name = 'ResnetGenerator'
        self.module_name = 'ResnetGeneratorModule'

    def __repr__(self) -> str:
        return 'ResnetGenerator'

    def get_num_blocks(self) -> int:
        return self.net.num_blocks

    def get_norm_type(self) -> str:
        return self.net.norm_type


class ResnetGenerator(nn.Module):
    def __init__(self, input_channels:int, output_channels:int, **kwargs) -> None:
        self.input_channels:int = input_channels
        self.output_channels:int = output_channels

        self.num_blocks:int      = kwargs.pop('num_blocks', 6)
        self.num_gen_filters:int = kwargs.pop('num_gen_filters', 64)
        self.drop_rate:float     = kwargs.pop('drop_rate', 0.0)
        self.norm_type:str       = kwargs.pop('norm_type', 'instance')

        if self.num_blocks <= 0:
            raise ValueError('num_blocks must be greater than 0')

        super(ResnetGeneratorModule, self).__init__()

        self.norm_layer = get_norm_layer(self.norm_type)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.model(X)

