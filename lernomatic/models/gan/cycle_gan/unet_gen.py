"""
UNET GENERATOR

Stefan Wong 2019
"""

import torch
import torch.nn as nn
from lernomatic.models import common


# Most of this is just adapted straight from
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class UNETSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nf:int, inner_nf:int, submodule=None, **kwargs) -> None:
        """
        UNETSkipConnectionBlock
        Implements a single skip connection in a UNet

        Arguments:

            outer_nf: (int)
                Number of filters to use in outer convolutional layer

            inner_nf: (int)
                Number of filters to use in inner convolutional layer

            outermost: (bool)
                Indicates if this is the outermost module (default: False)

            innermost: (bool)
                Indicates if this is the innermost module (default: False)

            input_num_channels: (int)
                Number of input channels (default: None)

            norm_layer: (nn.Module)
                Type of normalizatio layer to use. (default: nn.BatchNorm2d)

            ksize: (int)
                Kernel size (default: 4)

            stride: (int)
                Convolution stride (default: 1)

            pad_size: (int)
                Convolution padding size (default: 0)

        """
        self.outer_nf:int           = outer_nf
        self.inner_nf:int           = inner_nf
        self.outermost:bool         = kwargs.pop('outermost', False)
        self.innermost:bool         = kwargs.pop('innermost', False)
        self.input_num_channels:int = kwargs.pop('input_num_channels', None)
        self.norm_layer             = kwargs.pop('norm_layer', None)
        self.ksize:int              = kwargs.pop('ksize', 4)
        self.stride:int             = kwargs.pop('stride', 2)
        self.pad_size:int           = kwargs.pop('pad_size', 1)

        super(UNETSkipConnectionBlock, self).__init__()

        if self.norm_layer is None:
            self.norm_layer = nn.BatchNorm2d()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if self.input_num_channels is None:
            self.input_num_channels = self.outer_nf

        downconv = nn.Conv2d(
            self.input_num_channels,
            self.input_nf,
            kernel_size = self.ksize,
            stride      = self.stride,
            padding     = self.pad_size,
            bias        = use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = self.norm_layer(self.inner_nf)
        uprelu   = nn.ReLU(True)
        upnorm   = self.norm_layer(self.outer_nf)

        if self.outermost:
            upconv = nn.ConvTranspose2d(
                self.inner_nf * 2,
                self.outer_nf,
                kernel_size = self.ksize,
                stride = self.stride,
                padding = self.pad_size
            )
            down = [downconv]
            up   = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up

        elif self.innnermost:

            upconv = nn.ConvTranspose2d(
                self.inner_nf,
                self.outer_nf,
                kernel_size = self.ksize,
                stride = self.stride,
                padding = self.pad_size,
                bias = use_bias
            )
            down = [downrelu, downconv]
            up   = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(
                self.inner_nf * 2,
                self.outer_nf,
                kernel_size = self.ksize,
                stride = self.stride,
                padding = self.pad_size,
                bias = use_bias
            )
            down = [downrelu, downconv, downnorm]
            up   = [uprelu, upconv, upnorm]

            if self.use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(X)
        else:
            return torch.cat([X, self.model(X)], 1)



class UNETGenerator(common.LernomaticModel):
    """
    UNETGenerator
    LernomaticModel wrapper around a Generator based on the U-NET Architecture
    """
    def __init__(self, input_nc:int, output_nc:int, num_downsamples:int,
                 num_filters:int=64, **kwargs) -> None:

        self.net = UNETGeneratorModule(
            input_nc,
            output_nc,
            num_downsamples,
            num_filters = num_filters,
            **kwargs)

        self.import_path        : str = 'lernomatic.model.gan.cycle_gan.unet_gen'
        self.module_import_path : str = 'lernomatic.model.gan.cycle_gan.unet_gen'
        self.model_name         : str = 'UNETGenerator'
        self.module_name        : str = 'UNETGenenratorModule'

    def __repr__(self) -> str:
        return 'UNETGenerator (%d downsamples)' % self.net.num_downsamples

    def get_num_downsamples(self) -> int:
        return self.net.num_downsamples

    def get_params(self) -> dict:
        params = super(WideResnet, self).get_params()
        params['gen_params'] = {
            'input_nc'        : self.net.input_nc,
            'output_nc'       : self.net.output_nc,
            'num_downsamples' : self.net.num_downsamples,
            'num_filters'     : self.net.num_filters,
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
            params['gen_params']['input_nc'],
            params['gen_params']['output_nc'],
            num_filters     = params['gen_params']['num_filters'],
            num_downsamples = params['gen_params']['num_downsamples'],
        )
        self.net.load_state_dict(params['model_state_dict'])



class UNETGeneratorModule(nn.Module):
    """
    UNETGenerator
    Instantiates a collection of UNETSkipConnectionBlocks.
    """
    def __init__(self, input_nc:int, output_nc:int, num_downsamples:int,
                 num_filters:int=64, **kwargs) -> None:

        self.input_nc:int    = input_nc
        self.output_nc:int   = output_nc
        self.num_filters:int = num_filters
        self.num_downsamples = num_downsamples
        self.norm_layer      = kwargs.pop('norm_layer', None)
        super(UNETGeneratorModule, self).__init__()

        if self.norm_layer is None:
            self.norm_layer = nn.BatchNorm2d

        # generate the UNET structure
        unet_block = UNETSkipConnection(
            self.num_filters * 8,
            self.num_filters * 8,
            input_nc = None,
            submodule=None,
            innermost=True,
            norm_layer = self.norm_layer
        )
        for block in range(num_downsamples - 5):
            unet_block = UNETSkipConnection(
                self.num_filters * 8,
                self.num_filters * 8,
                input_nc = None,
                submodule = unet_block,
                norm_layer = self.norm_layer
            )
        # reduce the number of filters down from 8 * num_filters to num_filters
        unet_block = UNETSkipConenctionBlock(
            self.num_filters * 4,
            self.num_filters * 8,
            input_nc = None,
            submodule = unet_block,
            norm_layer = self.norm_layer
        )
        unet_block = UNETSkipConnectionBlock(
            self.num_filters * 2,
            self.num_filters * 4,
            input_nc = None,
            submodule = self.norm_layer
        )
        self.model = UNETSkipConnectionBlock(
            self.num_filters,
            self.num_filters * 2,
            input_nc = self.input_nc,
            outermost = True,
            submodule = self.norm_layer
        )

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.model(X)
