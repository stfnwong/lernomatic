"""
DENOISE_AE
Denoising Autoencoder. This is just a really simple denoising-autoencoder model,
probably won't be useful for anything that exciting.

Stefan Wong 2019
"""

import importlib
import torch
import torch.nn as nn
from lernomatic.models import common

from pudb import set_trace; set_trace()


# ======== ENCODER SIDE ======== #
class DAEEncoderBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,
                 kernel_size:int=3, stride:int=1, padding:int=1) -> None:
        super(DAEEncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.conv(X)
        out = self.relu(out)
        out = self.bn(out)

        return out


class DAEEncoderModule(nn.Module):
    """
    DAEEncoderModule

    TODO : Docstring
    """
    def __init__(self, **kwargs) -> None:
        self.num_blocks:int  = kwargs.pop('num_blocks', 4)
        self.start_size:int  = kwargs.pop('start_size', 32)
        self.kernel_size:int = kwargs.pop('kernel_size', 3)

        super(DAEEncoderModule, self).__init__()

        # "Input" half of encoder
        in_blocks = []
        cur_channel_size = self.start_size
        print('DAEEncoder')
        for n in range(self.num_blocks):
            if n == 0:
                block = DAEEncoderBlock(
                    1,
                    cur_channel_size,
                    kernel_size = self.kernel_size,
                )
            elif (n > 0) and (n % 2 != 0):
                block = DAEEncoderBlock(
                    cur_channel_size,
                    cur_channel_size,
                    kernel_size = self.kernel_size
                )
            else:
                block = DAEEncoderBlock(
                    cur_channel_size,
                    int(2 * cur_channel_size),
                    kernel_size = self.kernel_size
                )
                cur_channel_size = 2 * cur_channel_size

            in_blocks += [block]
            print(n, block.in_channels, block.out_channels)

        # "output" half. This is one block shorter than input size
        out_blocks = []
        for n in range(self.num_blocks - 1):
            if (n > 0) and (n % 2 != 0):
                block = DAEEncoderBlock(
                    cur_channel_size,
                    cur_channel_size,
                    kernel_size = self.kernel_size
                )
            else:
                block = DAEEncoderBlock(
                    cur_channel_size,
                    int(2 * cur_channel_size),
                    kernel_size = self.kernel_size
                )
                cur_channel_size = 2 * cur_channel_size

            out_blocks += [block]
            if n == (self.num_blocks - 2):
                out_blocks += [nn.MaxPool2d(2, 2)]

            print(n, block.in_channels, block.out_channels)

        self.in_blocks = nn.Sequential(*in_blocks)
        self.pool = nn.MaxPool2d(2, 2)
        self.out_blocks = nn.Sequential(*out_blocks)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.in_blocks(X)
        out = self.pool(out)
        out = self.out_blocks(out)
        #out = out.view(X.shape[0], -1)

        return out



# LernomaticModel implementation
class DAEEncoder(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.import_path       : str = 'lernomatic.models.autoencoder.denoise_ae'
        self.model_name        : str = 'DAEEncoder'
        self.module_name       : str = 'DAEEncoderModule'
        self.module_import_path: str = 'lernomatic.models.autoencoder.denoise_ae'
        self.net = DAEEncoderModule(**kwargs)

    def __repr__(self) -> str:
        return 'DAEEncoder'

    def get_num_blocks(self) -> int:
        return self.net.num_blocks

    def get_model_args(self) -> dict:
        return {
            'num_blocks'   : self.net.num_blocks,
            'start_size'   : self.net.start_size,
            'kernel_size'  : self.net.kernel_size
        }

    def set_params(self, params : dict) -> None:
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)

        self.net = mod(
            num_blocks  = params['model_args']['num_blocks'],
            start_size  = params['model_args']['start_size'],
            kernel_size = params['model_args']['kernel_size'],
        )
        self.net.load_state_dict(params['model_state_dict'])



# ======== DECODER SIDE ======== #
class DAEDecoderBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int,
                 kernel_size:int=3, stride:int=1, padding:int=1,
                 out_padding:int=0) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(DAEDecoderBlock, self).__init__()

        self.conv = nn.ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size= kernel_size,
            stride = stride,
            padding = padding,
            output_padding = out_padding
        )
        self.relu = nn.ReLU()
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.conv(X)
        out = self.relu(out)
        out = self.bn(out)

        return out



# check that start_size is a power of 2?
class DAEDecoderModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.num_blocks:int  = kwargs.pop('num_blocks', 4)
        self.kernel_size:int = kwargs.pop('kernel_size', 3)
        self.start_size:int  = kwargs.pop('start_size', 256)

        super(DAEDecoderModule, self).__init__()

        in_blocks = []
        cur_channel_size = self.start_size
        print('DAEDecoder')
        for n in range(self.num_blocks):
            if (n > 0) and (n % 2 != 0):
                block = DAEDecoderBlock(
                    cur_channel_size,
                    cur_channel_size,
                    kernel_size = self.kernel_size,
                )
            else:
                block = DAEDecoderBlock(
                    cur_channel_size,
                    int(cur_channel_size // 2),
                    kernel_size = self.kernel_size,
                    stride = 2 if (n == 0) else 1,
                    out_padding = 1 if (n == 0) else 0
                )
                cur_channel_size = cur_channel_size // 2

            in_blocks += [block]
            print(n, block.in_channels, block.out_channels)

        out_blocks = []
        for n in range(self.num_blocks-2):
            if (n > 0) and (n % 2 != 0):
                block = DAEDecoderBlock(
                    cur_channel_size,
                    cur_channel_size,
                    kernel_size = self.kernel_size
                )
            else:
                block = DAEDecoderBlock(
                    cur_channel_size,
                    int(cur_channel_size // 2),
                    kernel_size = self.kernel_size
                )
                cur_channel_size = cur_channel_size // 2
            out_blocks += [block]
            print(n, block.in_channels, block.out_channels)


        final_block = [
            nn.ConvTranspose2d(
                cur_channel_size,
                1,
                kernel_size = self.kernel_size,
                stride = 2,
                padding = 1,
                output_padding = 1
            ),
            nn.ReLU()
        ]
        print(n+1, cur_channel_size, 1)

        self.in_blocks   = nn.Sequential(*in_blocks)
        self.out_blocks  = nn.Sequential(*out_blocks)
        self.final_block = nn.Sequential(*final_block)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        #out = X.view(X.shape[0], self.start_size, -1)
        out = self.in_blocks(X)
        out = self.out_blocks(out)
        out = self.final_block(out)

        return out



# LernomaticModel implementation
class DAEDecoder(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.import_path       : str = 'lernomatic.models.autoencoder.denoise_ae'
        self.model_name        : str = 'DAEDecoder'
        self.module_name       : str = 'DAEDecoderModule'
        self.module_import_path: str = 'lernomatic.models.autoencoder.denoise_ae'
        self.net = DAEDecoderModule(**kwargs)

    def __repr__(self) -> str:
        return 'DAEDecoder'

    def get_num_blocks(self) -> int:
        return self.net.num_blocks

    def get_model_args(self) -> dict:
        return {
            'num_blocks'   : self.net.num_blocks,
            'start_size'   : self.net.start_size,
            'kernel_size'  : self.net.kernel_size
        }

    def set_params(self, params : dict) -> None:
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)

        self.net = mod(
            num_blocks  = params['model_args']['num_blocks'],
            start_size  = params['model_args']['start_size'],
            kernel_size = params['model_args']['kernel_size'],
        )
        self.net.load_state_dict(params['model_state_dict'])
