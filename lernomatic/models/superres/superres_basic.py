"""
SUPERRES_BASIC
Some common modules for superresolution nets

Stefan Wong 2019
"""

import math
import torch
import torch.nn as nn

def superres_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding = (kernel_size // 2),
        bias = bias
    )


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, **kwargs):
        self.rgb_range = kwargs.pop('rgb_range', (0.4488, 0.4371, 0.4040))
        self.rgb_std   = kwargs.pop('rgb_std',   (1.0, 1.0, 1.0))
        self.sign      = kwargs.pop('sign', -1)

        super(MeanShift, self).__init__()
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data   = self.sign * self.rgb_range * torch.Tensor(self.rgb_mean) / std

        for p in self.parameters():
            p.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv_module, scale, num_features, **kwargs):
        self.do_batchnorm = kwargs.pop('do_batchnorm', False)
        self.activation   = kwargs.pop('activation', None)
        self.bias         = kwargs.pop('bias', False)

        modules = []
        if(scale & (scale -1)) == 0:       # is scale 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv_module(
                    num_features,
                    4 * num_features,
                    3,
                    self.bias)
                )
                modules.append(nn.PixelShuffle(2))
                if self.do_batchnorm:
                    modules.append(nn.Batchnorm2d(num_features))
                if self.activation == 'relu':
                    modules.append(nn.ReLU(True))
                elif self.activation == 'prelu':
                    modules.append(nn.PReLU(num_features))

        elif scale == 3:
            modules.append(conv_module(
                num_features,
                9 * num_features,
                3,
                self.bias)
            )
            if self.do_batchnorm:
                modules.append(nn.Batchnorm2d(num_features))
            if self.activation == 'relu':
                modules.append(nn.ReLU(True))
            elif self.activation == 'prelu':
                modules.append(nn.PReLU(num_features))

        else:
            raise NotImplementedError

        super(Upsampler, self).__init__()
