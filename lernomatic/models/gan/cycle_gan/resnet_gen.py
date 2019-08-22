"""
RESNET_GENERATOR
A Resnet-based generator module made from a few resnet blocks
and some downsampling operations.

Adapted in large part from models in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Stefan Wong 2019
"""

import importlib
import functools
import torch
import torch.nn as nn
from lernomatic.models import common
from lernomatic.models.gan import gan_norm


# This is the same resnet block as in
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, num_channels:int, pad_type:str, use_dropout:bool,
                 use_bias:bool, norm_layer=nn.BatchNorm2d, ksize:int=3) -> None:
        valid_pad_types = ('reflect', 'replicate', 'zero')
        super(ResnetBlock, self).__init__()

        if pad_type not in valid_pad_types:
            raise NotImplementedError('Pad type [%s] not implemented, must be one of %s' %\
                        (str(pad_type), str(valid_pad_types))
            )

        block = []
        p = 0
        if pad_type == 'reflect':
            block += [nn.ReflectionPad2d(1)]
        elif pad_type == 'replicate':
            block += [nn.ReplicationPad2d(1)]
        elif pad_type == 'zero':
            p = 1

        block += [nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size = ksize,
                padding = p,
                bias = use_bias
            ),
            norm_layer(num_channels),
            nn.ReLU(True)
        ]

        if use_dropout:
            block += [nn.Dropout(0.5)]

        p = 0
        if pad_type == 'reflect':
            block += [nn.ReflectionPad2d(1)]
        elif pad_type == 'replicate':
            block += [nn.ReplicationPad2d(1)]
        elif pad_type == 'zero':
            p = 1

        block += [nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size = ksize,
                padding = p,
                bias = use_bias
            ),
            norm_layer(num_channels)
        ]

        self.conv_block = nn.Sequential(*block)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = X + self.conv_block(X)
        return out



class ResnetGenerator(common.LernomaticModel):
    def __init__(self,
                 num_input_channels:int=3,
                 num_output_channels:int=3,
                 **kwargs) -> None:
        self.net = ResnetGeneratorModule(
            num_input_channels,
            num_output_channels,
            **kwargs)
        self.import_path        = 'lernomatic.models.gan.cycle_gan.resnet_gen'
        self.module_import_path = 'lernomatic.models.gan.cycle_gan.resnet_gen'
        self.model_name         = 'ResnetGenerator'
        self.module_name        = 'ResnetGeneratorModule'

    def __repr__(self) -> str:
        return 'ResnetGenerator'

    def get_num_blocks(self) -> int:
        return self.net.num_blocks

    def get_norm_type(self) -> str:
        return self.net.norm_type

    def get_params(self) -> dict:
        params = super(ResnetGenerator, self).get_params()
        params['gen_params'] = {
            'num_input_channels' : self.net.input_channels,
            'num_output_channels': self.net.output_channels,
            'num_filters'        : self.net.num_filters,
            'num_blocks'         : self.net.num_blocks,
            'norm_type'          : self.net.norm_type,
            'padding_type'       : self.net.padding_type,
            'use_dropout'        : self.net.use_dropout,
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
            params['gen_params']['num_input_channels'],
            params['gen_params']['num_output_channels'],
            num_filters     = params['gen_params']['num_filters'],
            num_blocks      = params['gen_params']['num_blocks'],
            norm_type       = params['gen_params']['norm_type'],
            padding_type    = params['gen_params']['padding_type'],
            use_dropout     = params['gen_params']['use_dropout'],
        )
        self.net.load_state_dict(params['model_state_dict'])



class ResnetGeneratorModule(nn.Module):
    def __init__(self, input_channels:int, output_channels:int, **kwargs) -> None:
        self.input_channels:int = input_channels
        self.output_channels:int = output_channels

        self.num_blocks:int      = kwargs.pop('num_blocks', 6)
        self.num_filters:int     = kwargs.pop('num_filters', 64)
        self.drop_rate:float     = kwargs.pop('drop_rate', 0.0)
        self.norm_type:str       = kwargs.pop('norm_type', 'instance')
        self.padding_type:str    = kwargs.pop('padding_type', 'reflect')
        self.use_dropout:bool    = kwargs.pop('use_dropout', False)

        if self.num_blocks <= 0:
            raise ValueError('num_blocks must be greater than 0')

        super(ResnetGeneratorModule, self).__init__()

        self.norm_layer = gan_norm.get_norm_layer(self.norm_type)
        if type(self.norm_layer) == functools.partial:
            use_bias = self.norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = self.norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                self.input_channels,
                self.num_filters,
                kernel_size = 7,
                padding = 0,
                bias = use_bias
            ),
            self.norm_layer(self.num_filters),
            nn.ReLU(True)
        ]

        # add downsampling layers
        num_ds_layers = 2
        for l in range(num_ds_layers):
            mult = 2 ** l
            model += [
                nn.Conv2d(
                    self.num_filters * mult,
                    self.num_filters * mult * 2,
                    kernel_size = 3,
                    stride = 2,
                    padding = 1,
                    bias = use_bias
                ),
                self.norm_layer(self.num_filters * mult * 2),
                nn.ReLU(True)
            ]

        # add resnet blocks
        mult = 2 ** num_ds_layers
        for l in range(self.num_blocks):
            model += [
                ResnetBlock(
                    self.num_filters * mult,
                    pad_type = self.padding_type,
                    norm_layer = self.norm_layer,
                    use_dropout = self.use_dropout,
                    use_bias = use_bias
                )
            ]

        # add upsampling layers
        for l in range(num_ds_layers):
            mult = 2 ** (num_ds_layers - l)
            model += [
                nn.ConvTranspose2d(
                    self.num_filters * mult,
                    int(self.num_filters * mult / 2),
                    kernel_size = 3,
                    stride = 2,
                    padding = 1,
                    output_padding = 1,
                    bias = use_bias
                ),
                self.norm_layer(int(self.num_filters * mult / 2)),
                nn.ReLU(True)
            ]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(self.num_filters, self.output_channels, kernel_size = 7, padding = 0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.model(X)
