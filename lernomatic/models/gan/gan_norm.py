"""
GAN_NORM
Get normalization layers for a GAN model

Stefan Wong 2019
"""

import functools
import torch
import torch.nn as nn


def get_norm_layer(norm_type:str='instance') -> nn.Module:

    if norm_type == 'batch':
        norm_layer = functools.partial(
            nn.BatchNorm2d,
            affine=True,
            track_running_stats=True
        )
    elif norm_type == 'instance':
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
