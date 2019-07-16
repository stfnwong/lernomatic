"""
GAN_RESNET
GAN-specific implementation of Resnet blocks
Main difference is addition of padding block options

Stefan Wong 2019
"""

import torch
import torch.nn as nn


class GANResnetBlock(nn.Module):
    # TODO: what to do about norm layer?
    def __init__(self,
                 conv_size:int,
                 drop_rate:float,
                 use_bias:bool,
                 pad_type:str='reflect',
                 **kwargs
                 ) -> None:
        super(GANResnetBlock, self).__init__()
        self.kernel_size:int = kwargs.pop('kernel_size', 3)

        conv_block = []
        p = 0
        if pad_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif pad_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif pad_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('Pad type [%s] not implemented' % str(pad_type))

        conv_block += [nn.Conv2d(
            conv_size,
            conv_size,
            kernel_size=self.kernel_size,

    def foward(self, X:torch.Tensor) -> torch.Tensor:
        out = X + self.conv_block(X)
        return out
