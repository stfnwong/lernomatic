"""
SUPERRES_RESNET
These are based on the networks in https://arxiv.org/abs/1707.02921

Stefan Wong 2019
"""

import torch
import torch.nn as nn
from lernomatic.models.superres import superres_basic
from lernomatic.models import resnet

# debug
#from pudb import set_trace; set_trace()



# TODO : defaults for rgb_range?
class MDSR(nn.Module):
    def __init__(self, rgb_range, conv_module=None, **kwargs):
        # TODO : check sensible defaults
        self.num_res_blocks = kwargs.pop('num_res_blocks', 28)
        self.num_features   = kwargs.pop('num_features', 256)
        self.kernel_size    = kwargs.pop('kernel_size', 3)
        self.activation     = kwargs.pop('activation', 'relu')
        self.num_colors     = kwargs.pop('num_colors', 3)
        self.rgb_range      = kwargs.pop('rgb_range', 255)      # TODO : check that this is 24bpp range
        self.scale          = kwargs.pop('scale', [4, 4])       # TODO : check the form that this arg should take

        if conv_module is None:
            self.conv = superres_basic.superres_conv

        super(MDSR, self).__init__()
        # meanshift modules
        self.sub_mean = superres_basic.MeanShift(
            self.rgb_range
        )
        self.add_mean = superres_basic.MeanShift(
            self.rgb_range,
            sign=1
        )

        net_head = [conv_module(
            self.num_colors,
            self.num_features,
            self.kernel_size)
        ]

        self.pre_process = nn.ModuleList([
            nn.Sequential(
                superres_basic.SRResBlock(
                    self.conv(
                        self.num_features,
                        5,
                        nn.ReLU(True))
                )
           ) for l in range(self.num_res_blocks)
        ])

        net_body = [
            superres_basic.SRResBlock(
                conv_module,
                self.num_features,
                5,
                act=self.activation
            ) for n in range(self.num_res_blocks)
        ]
        net_body.append(conv_module(
            self.num_features,
            self.num_features,
            self.kernel_size)
        )

        self.upsample = nn.ModuleList([superres_basic.Upsampler(
            conv_module,
            s,
            self.num_features,
            act=False) for s in self.scale
        ])

        net_tail = [conv_module(self.num_features, self.num_colors, self.kernel_size)]

        self.head = nn.Sequential(*net_head)
        self.body = nn.Sequential(*net_body)
        self.tail = nn.Sequential(*net_tail)

    def forward(self, X):
        out = self.sub_mean(X)
        out = self.head(out)
        out = self.pre_process[self.scale_idx](out)

        res = self.body(X)
        res += X

        u = self.upsample[self.scale_idx](res)
        u = self.tail(u)
        u = self.add_mean(u)

        return u

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
