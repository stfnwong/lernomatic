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
from lernomatic.models.gan import lrgan_functions
from lernomatic.util import math_util



# Grid generation
class AffineGridGen(nn.Module):
    def __init__(self, height:int, width:int, lr:int = 1, aux_loss:bool=False) -> None:
        self.height = height
        self.width = width
        self.aux_loss = aux_loss
        self.lr = lr
        self.f = AffineGridGenFunction(self.height, self.width, lr=self.lr)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        if not self.aux_loss:
            return self.f(X)

        # TODO : make dtype settable
        identity = torch.from_numpy(np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32))
        batch_identity = torch.zeros([X.size(0), 2, 3])
        # TODO : what is going on here
        for i in range(X.size(0)):
            batch_identity[i] = identity

        loss = torch.mul(X - batch_identity, X - batch_identity)
        loss = torch.sum(loss, 1)
        loss = torch.sum(loss, 2)

        return self.f(X), loss.view(-1, 1)



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

    def get_zvec_dim(self) -> int:
        return self.net.zvec_dim

    def get_num_filters(self) -> int:
        return self.net.num_filters

    def get_num_timesteps(self) -> int:
        return self.net.num_timesteps


# ==== BG Generator ==== #
class LRGANBGGen(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.zvec_dim    :int = kwargs.pop('zvec_dim', 128)
        self.num_filters :int = kwargs.pop('num_filters' 64)
        self.kernel_size :int = kwargs.pop('kernel_size', 4)
        self.img_size    :int = kwargs.pop('img_size', 64)

        super(LRGANBGGen, self).__init__()

        if not math_util.is_pow2_int(self.img_size):
            raise ValueError('Image size [%d] must be a power of 2' % str(self.img_size))

        # build the network
        self.start_conv = nn.ConvTranspose2d(
            self.zvec_dim,
            self.num_filters * 4,
            kernel_size = self.kernel_size
            stride = 4,
            padding = 0,
            bias = True
        )
        self.start_bn = nn.BatchNorm2d(self.num_filters * 4)
        self.start_relu = nn.ReLU(inplace=True)

        # Add the rest of the modules based on the required image size
        gen_blocks = []
        cur_size = 4
        conv_in_depth = 4 * self.num_filters
        conv_out_depth = 2 * self.num_filters

        while cur_size < (self.img_size // 2):
            gen_blocks += [
                nn.ConvTranspose2d(
                    conv_in_depth,
                    conv_out_depth,
                    kernel_size = self.kernel_size,
                    stride = 2,
                    padding = 1,
                    bias = True
                )
            ]
            gen_blocks += [
                nn.BatchNorm2d(conv_depth_out)
            ]
            gen_blocks += [
                nn.ReLU(True)
            ]

            print('LRGANBGGen Generated module for size %d, conv_depth_in: %d, conv_depth_out: %d' %\
                  (cur_size, conv_depth_in, conv_depth_out)
            )
            conv_depth_in = conv_depth_out
            conv_depth_out = max(conv_depth_in // 2, 64)
            cur_size = cur_size * 2

        self.net = nn.Sequential(*gen_blocks)
        self.final_depth_in = depth_in

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.start_conv(X)
        out = self.start_bn(out)
        out = self.start_relu(out)
        out = self.net(out)

        return out



# ==== Mask Generator ==== #
class LRGANFGGen(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.zvec_dim : int = kwargs.pop('zvec_dim', 128)
        self.num_filters:int = kwargs.pop('num_filters' 64)
        self.kernel_size:int = kwargs.pop('kernel_size', 4)
        self.img_size :int = kwargs.pop('img_size', 64)

        super(LRGANFGGen, self).__init__()

        # build the network
        self.start_conv = nn.ConvTranspose2d(
            self.zvec_dim,
            self.num_filters * 8,
            kernel_size = self.kernel_size,
            stride = 4,
            padding = 4,
            output_padding = 0,
            bias = True
        )
        self.start_bn = nn.Batchnorm2d(self.num_filters * 8)
        self.start_relu = nn.ReLU(inplace=True)

        gen_blocks = []
        cur_size = 4
        conv_depth_in = 8 * self.num_filters
        conv_depth_out = 4 * self.num_filters

        while cur_size < (self.img_size // 2):
            gen_blocks += [
                nn.ConvTranspose2d(
                    conv_depth_in,
                    conv_depth_out,
                    kernel_size = self.kernel_size,
                    stride = 2,
                    padding = 1,
                    bias = False)
            ]
            gen_blocks += [
                nn.BatchNorm2d(conv_depth_out)
            ]
            gen_blocks += [
                nn.ReLU(inplace=True)
            ]

            print('LRGANFGGEn Generated module for size %d, conv_depth_in: %d, conv_depth_out: %d' %\
                  (cur_size, conv_depth_in, conv_depth_out)
            )

            conv_depth_in = conv_depth_out
            conv_depth_out = max(conv_depth_in // 2, 64)
            cur_size = cur_size * 2

        self.net = nn.Sequential(*gen_blocks)
        self.final_depth_in = depth_in

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.start_conv(X)
        out = self.start_bn(out)
        out = self.start_relu(out)
        out = self.net(out)

        return out


# ======== FC Encoder ======== #
class LRGANEncoderFC(nn.Module):
    def __init__(self, **kwargs) -> None:
        # TODO : these default numbers make no sense
        self.depth_in:int = kwargs.pop('depth_in', 64)
        self.depth_out:int = kwargs.pop('depth_out', 64)
        self.nsize_in:int = kwargs.pop('nsize_in', 64)

        super(LRGANEncoderFC, self).__init__()

        self.fc1 = nn.Linear(
            self.depth_in * self.nsize_in * self.nsize_in,
            self.depth_out
        )
        self.bn = nn.BatchNorm1d(self.depth_out)
        self.tan = nn.Tanh()

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.fc1(X)
        out = self.bn(out)
        out = self.tan(out)

        return out


# ======== Conv Encoder ======== #
class LRGANEncoderConv(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.depth_in:int = kwargs.pop('depth_in', 64)
        self.depth_out:int = kwargs.pop('depth_out', 64)
        self.nsize_in:int = kwargs.pop('nsize_in', 64)

        super(LRGANEncoderConv, self).__init__()

        # Create a cascade of convolutions
        cur_depth = self.nsize_in
        conv_blocks = []
        while cur_depth > self.depth_out:
            conv_blocks += [
                nn.AvgPool2d(4, 2, 1)
            ]
            conv_blocks += [
                nn.BatchNorm2d(self.depth_in)
            ]
            conv_blocks += [
                nn.LeakyReLU(0.2, inplace=True)
            ]
            cur_depth = cur_depth // 2

        self.net = nn.Sequential(*conv_blocks)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.net(X)

        return out



# ==== Complete generator module
class LRGANGeneratorModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.zvec_dim     :int = kwargs.pop('zvec_dim', 128)
        self.num_filters  :int = kwargs.pop('num_filters', 64)       # num generator features?
        self.num_timesteps:int = kwargs.pop('num_timesteps', 3)
        self.nsize_out    :int = kwargs.pop('nsize_out', 2)
        self.kernel_size  :int = kwargs.pop('kernel_size', 4)
        self.max_obj_scale:float = kwargs.pop('max_obj_scale', 1.2)     # maximum relative size of object to image

        super(LRGANGeneratorModule, self).__init__()

        # construct the network
        self.lstm = nn.LSTMCell(self.zvec_dim, self.zvec_dim)

        # Background image processor
        self.bg_gen = LRGANBGGen(
            depth_in = 0,
            depth_out = 0,
            nsize_in = 0
        )
        bg_i_modules = []
        bg_i_modules += [nn.ConvTranspose2d(
                self.bg_gen.final_depth_in,
                self.num_channels,
                kernel = self.kernel_size,
                stride = 2,
                padding = 1,
                bias = True
            )
        ]
        bg_i_modules += [nn.Tanh()]
        self.bg_gen_i = nn.Sequential(*bg_i_modules)

        # Foreground image processor
        self.fg_gen = LRGANFGGen(
            depth_in = 0,
            depth_out = 0,
            nsize_in = 0
        )
        fg_i_modules = []
        fg_i_modules += [nn.ConvTranspose2d(
                self.fg_gen.final_depth_in,
                self.num_channels,
                kernel = self.kernel_size,
                stride = 2,
                padding = 1,
                bias = True
            )
        ]
        self.fg_gen_i = nn.Sequential(*fg_i_modules)

        # Mask generator
        mask_gen_modules = []
        mask_gen_modules += [
            nn.Conv2dTranspose(
                self.fg_gen.final_depth_in,
                1,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = True
            )
        ]
        mask_gen_modules += [nn.Sigmoid()]
        self.mask_gen = nn.Sequential(*mask_gen_modules)

        # Generator Grid
        # linear layer for 6-d transform
        self.g_transform = nn.Linear(self.zvec_dim, 6)
        self._init_g_transform()

        # actual grid
        self.g_grid = AffineGridGen(self.img_size, self.img_size, aux_loss = False)

        # get some compositors
        self.compositors = []
        for t in range(self.num_timesteps-1):
            self.compositors.append(lrgan_functions.STNM())

        # convolutional encoder
        self.enc_conv = LRGANEncoderConv(
            self.fgnet.final_depth_in,
            self.img_size // 2,
            self.nsize_out
        )
        # linear encoder
        self.enc_fc = LRGANEncoderFC(
            self.fgnet.final_depth_in,
            self.nsize_out,
            self.zvec_dim
        )

        # final linear net
        nlnet_modules = []
        nlnet_modules += [
            nn.Linear(2 * self.zvec_dim, self.zvec_dim)
        ]
        nlnet_modules += [
            nn.BatchNorm1d(self.zvec_dim)
        ]
        nlnet_modules += [nn.Tang()]
        self.nlnet = nn.Sequential(*nlnet_modules)


    def _init_g_transform(self) -> None:
        self.g_transform.weight.data.zero_()
        self.g_transform.bias.data.zero_()
        self.g_transform.bias.data[0] = self.max_obj_scale
        self.g_transform.bias.data[4] = self.max_obj_scale


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

    def get_num_filters(self) -> int:
        return self.net.num_filters

    def zero_grad(self) -> None:
        self.net.zero_grad()

    def get_params(self) -> dict:
        params = super(DCGANDiscriminator, self).get_params()
        params['disc_params'] = {
            'num_filters'  : self.net.num_filters,
            'num_channels' : self.net.num_channels,
            'kernel_size'  : self.net.kernel_size,
            'img_size'     : self.net.img_size
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
            num_filters  = params['disc_params']['num_filters'],
            num_channels = params['disc_params']['num_channels'],
            kernel_size  = params['disc_params']['kernel_size'],
            img_size     = params['disc_params']['img_size'],
        )
        self.net.load_state_dict(params['model_state_dict'])



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

