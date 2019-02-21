"""
DISC_BLOCKS
Various discriminator blocks

"""

import functools
import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    """
    NLayerDiscriminator
    Defines a PatchGAN discriminator

    """
    def __init__(self, num_input_channels, num_filters=64, num_layers=3, norm_layer=nn.BatchNorm2d, **kwargs):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func != nn.BatchNorm2d)
        else:
            use_bias = (norm_layer != nn.BatchNorm2d)

        kw = kwargs.pop('kw', 4)
        padw = kwargs.pop('padw', 1)
        sequence = [
            nn.Conv2d(
                num_input_channels,
                num_filters,
                kernel_size=kw,
                stride=2,
                padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        # gradually increase the number of filters
        for n in range(1, num_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(num_filters + nf_mult_prev,
                          num_filters * nf_mult,
                          kernel_size = kw,
                          stride=2,
                          padding=padw,
                          bias=use_bias
                          ),
                norm_layer(num_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        sequence += [
            nn.Conv2d(num_filters * nf_mult_prev,
                      num_filters * nf_mult,
                      kernel_size = kw,
                      stride = 2,
                      padding = padw,
                      bias = use_bias
                      ),
            norm_layer(num_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            # output 1-channel prediction map
            nn.Conv2d(num_filters * nf_mult,
                      1,
                      kernel_size = kw,
                      stride = 1,
                      padding = padw
            )
        ]

        self.model = nn.Sequential(sequence)

    def forward(self, X):
        return self.model(X)
