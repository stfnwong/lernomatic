"""
GAN_LOSS

"""

import functools
import torch
import torch.nn as nn


VALID_GAN_MODES = ('lsgan', 'vanilla', 'wgangp')

# This is taken more or less directly from junyanz CycleGAN code
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class GANLoss(nn.Module):
    """
    GANLoss
    Loss with automatic sizing of target tensor.

    Arguments:
        gan_mode: (str)
            GAN objective. Must be one of 'lsgan', 'vanilla', or 'wgangp'
        target_real_label: (float)
            Label for a real image (default: 1.0)
        target_fake_label: (float)
            Label for a fake image (default: 0.0)
    """
    def __init__(self,
                 mode:str,
                 device,
                 target_real_label:float=1.0,
                 target_fake_label:float=0.0) -> None:
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = mode
        self.device = device
        if self.gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('GAN mode [%s] not implemented' % str(self.gan_mode))

    def __call__(self, pred:torch.Tensor, target_real:bool) -> torch.Tensor:
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(pred, target_real)
            loss = self.loss(pred, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_real:
                loss = -pred.mean()
            else:
                loss = pred.mean()

        return loss

    def get_target_tensor(self, pred:torch.Tensor, target_real:bool) -> torch.Tensor:
        """
        Create label tensors with same size as input
        """
        if target_real:
            target_tensor = self.real_label.to(self.device)
        else:
            target_tensor = self.fake_label.to(self.device)
        return target_tensor.expand_as(pred)
