"""
NLAYER_DISC
NLayer Discriminator Model

Stefan Wong 2019
"""

import functools
import torch
import torch.nn as nn
from lernomatic.models import common


class NLayerDiscriminator(common.LernomaticModel):
    def __init__(self,
                 num_input_channels:int,
                 num_filters:int=64,
                 num_layers:int=3,
                 **kwargs) -> None:
        self.net = NNLayerDiscriminatorModule(
            num_input_channels,
            num_filters=num_filtes,
            num_layers = num_layers,
            **kwargs)
        self.import_path        : str = 'lernomatic.models.gan.cycle_gan.nlayer_disc'
        self.module_import_path : str = 'lernomatic.models.gan.cycle_gan.nlayer_disc'
        self.model_name         : str = 'NLayerDiscriminator'
        self.module_name        : str = 'NLayerDiscriminatorModule'

    def __repr__(self) -> str:
        return 'NLayerDiscriminator (%d layers)' % self.net.num_layers

    def get_num_layers(self) -> int:
        return self.net.num_layers

    def get_num_filters(self) -> int:
        return self.net.num_filters

    def get_params(self) -> dict:
        params = super(WideResnet, self).get_params()
        params['disc_params'] = {
            'num_filters'        : self.net.num_filters,
            'num_input_channels' : self.net.num_input_channels,
            'num_layers'         : self.net.num_layers,
            'ksize'              : self.net.ksize,
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
            pad_size    = params['disc_params']['pad_size']
        )
        self.net.load_state_dict(params['model_state_dict'])



# Should kernel sizes, etc, be settable?
class NLayerDiscriminatorModule(nn.Module):
    """
    NLayerDiscriminatorModule
    Defines a PatchGAN discriminator

    """
    def __init__(self,
                 num_input_channels:int,
                 num_filters:int=64,
                 num_layers:int=3,
                 **kwargs) -> None:

        self.num_input_channels :int = num_input_channels
        self.num_filters        :int = num_filters
        self.num_layers         :int = num_layers
        self.ksize              :int = kwargs.pop('ksize', 4)       # kernel width
        self.pad_size           :int = kwargs.pop('pad_size', 1)
        self.norm_layer              = kwargs.pop('norm_layer', None)

        super(NLayerDiscriminator, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func != nn.BatchNorm2d)
        else:
            use_bias = (norm_layer != nn.BatchNorm2d)

        sequence = [
            nn.Conv2d(
                num_input_channels,
                num_filters,
                kernel_size=self.ksize,
                stride=2,
                padding=self.pad_size),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        # gradually increase the number of filters
        for n in range(1, self.num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(self.num_filters + nf_mult_prev,
                          self.num_filters * nf_mult,
                          kernel_size = self.ksize,
                          stride=2,
                          padding=self.pad_size,
                          bias=use_bias
                          ),
                norm_layer(self.num_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** self.num_layers, 8)
        sequence += [
            nn.Conv2d(self.num_filters * nf_mult_prev,
                      self.num_filters * nf_mult,
                      kernel_size = self.ksize,
                      stride = 2,
                      padding = self.pad_size,
                      bias = use_bias
                      ),
            norm_layer(self.num_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            # output 1-channel prediction map
            nn.Conv2d(self.num_filters * nf_mult,
                      1,
                      kernel_size = self.ksize,
                      stride = 1,
                      padding = self.pad_size
            )
        ]

        self.model = nn.Sequential(sequence)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.model(X)
