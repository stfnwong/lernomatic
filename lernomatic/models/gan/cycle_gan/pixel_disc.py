"""
PIXEL_DISC
PixelGAN Discriminator (1x1 PatchGAN )

Stefan Wong 2019
"""

import importlib
import functools
import torch
import torch.nn as nn
from lernomatic.models import common


class PixelDiscriminator(common.LernomaticModel):
    def __init__(self,
                 num_input_channels:int,
                 num_filters:int,
                 **kwargs) -> None:
        self.net = PixelDiscriminatorModule(
            num_input_channels,
            num_filters = num_filters,
            **kwargs)

        self.import_path        : str = 'lernomatic.models.gan.cycle_gan.pixel_disc'
        self.module_import_path : str = 'lernomatic.models.gan.cycle_gan.pixel_disc'
        self.model_name         : str = 'PixelDiscriminator'
        self.module_name        : str = 'PixelDiscriminatorModule'

    def __repr__(self) -> str:
        return 'PixelDiscriminator'

    def get_num_filters(self) -> int:
        return self.net.num_filters

    def get_params(self) -> dict:
        params = super(PixelDiscriminator, self).get_params()
        params['disc_params'] = {
            'num_filters'        : self.net.num_filters,
            'num_input_channels' : self.net.num_input_channels,
            'ksize'              : self.net.ksize,
            'stride'             : self.net.stride,
            'pad_size'           : self.net.pad_size
        }

        return params

    def set_params(self, params : dict) -> None:
        # regular model stuff
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)
        self.net = mod(
            params['disc_params']['num_input_channels'],
            num_filters = params['disc_params']['num_filters'],
            num_layers  = params['disc_params']['num_layers'],
            ksize       = params['disc_params']['ksize'],
            stride      = params['disc_params']['stride'],
            pad_size    = params['disc_params']['pad_size']
        )
        self.net.load_state_dict(params['model_state_dict'])



class PixelDiscriminatorModule(nn.Module):
    def __init__(self,
                 num_input_channels:int,
                 num_filters:int=64,
                 **kwargs) -> None:

        self.num_input_channels:int = num_input_channels
        self.num_filters:int    = num_filters
        self.ksize:int          = kwargs.pop('ksize', 1)
        self.stride:int         = kwargs.pop('stride', 1)
        self.pad_size:int       = kwargs.pop('pad_size', 0)
        self.norm_layer         = kwargs.pop('norm_layer', None)
        super(PixelDiscriminatorModule, self).__init__()

        if self.norm_layer is None:
            self.norm_layer = nn.BatchNorm2d

        if type(self.norm_layer) == functools.partial:
            use_bias = self.norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = self.norm_layer != nn.InstanceNorm2d

        self.model = [
            nn.Conv2d(self.num_input_channels,
                      self.num_filters,
                      kernel_size = self.ksize,
                      stride = self.stride,
                      padding = self.pad_size
            ),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.num_filters,
                      self.num_filters * 2,
                      kernel_size = self.ksize,
                      stride = self.stride,
                      padding = self.pad_size,
                      bias = use_bias
            ),
            self.norm_layer(self.num_filters * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.num_filters * 2,
                      1,
                      kernel_size = self.ksize,
                      stride = self.stride,
                      padding = self.pad_size,
                      bias = use_bias
            )
        ]
        self.model = nn.Sequential(*self.model)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.model(X)
