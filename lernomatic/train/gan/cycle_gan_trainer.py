"""
CYCLE_GAN_TRAINER
Trainer for Cycle GAN model

Stefan Wong 2019
"""

import time
import torch
import torch.nn as nn
from lernomatic.models import common
from lernomatic.train import trainer

# debug
from pudb import set_trace; set_trace()


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
                 target_real_label:float=1.0,
                 target_fake_label:float=0.0) -> None:
        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.Tensor(target_real_label))
        self.register_buffer('fake_label', torch.Tensor(target_fake_label))

        self.gan_mode = mode
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
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(pred)


class CycleGANTrainer(trainer.Trainer):
    """
    CycleGANTrainer
    Trainer object for CycleGAN experiments

    """
    def __init(self,
               gen_ab: common.LernomaticModel=None,
               gen_ba: common.LernomaticModel=None,
               disc_a: common.LernomaticModel=None,
               disc_b: common.LernomaticModel=None,
               **kwargs) -> None:
        self.gen_ab = gen_ab
        self.gen_ba = gen_ba
        self.disc_a = disc_a
        self.disc_b = disc_b

        self.beta1:float = kwargs.pop('beta1', 1.0)     # TODO: what is a good default for this?
        super(CycleGANTrainer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'CycleGANTrainer'

    def _init_optimizer(self) -> None:
        # here I will use the same 'chained' optimizer structure as in junyanz
        # Generator side optimizers
        if self.gan_ab is not None and self.gan_ba is not None:
            self.optim_gen = torch.optim.Adam(
                itertools.chain(self.gan_ab.get_model_parameters(), self.gan_ba.get_model_parameters()),
                lr=self.learning_rate,
                betas = (self.beta1, 0.999)
            )
        else:
            self.optim_gen = None

        # Discriminator side optimizers
        if self.disc_a is not None and self.disc_b is not None:
            self.optim_disc = torch.optim.Adam(
                itertools.chain(self.disc_a.get_model_parameters(), self.disc_b.get_model_parameters()),
                lr=self.learning_rate,
                betas = (self.beta1, 0.999)
            )
        else:
            self.optim_disc = None

    def _init_dataloaders(self) -> None:
        pass

    def train_epoch(self):

        for n, (data, labels) in enumerate(self.train_loader):
            pass



    # TODO : checkpoint, history, etc
