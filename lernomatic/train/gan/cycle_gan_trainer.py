"""
CYCLE_GAN_TRAINER
Trainer for Cycle GAN model

Stefan Wong 2019
"""

import time
import torch
import torch.nn as nn
from lernomatic.models.gan import gan_loss
from lernomatic.models import common
from lernomatic.train import trainer

# debug
#from pudb import set_trace; set_trace()



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

        # TODO : init criterion here?

    def _init_dataloaders(self) -> None:
        pass

    def train_epoch(self):

        for n, (a_img, b_img) in enumerate(self.train_loader):

            a_img = a_img.to(self.device)
            b_img = b_img.to(self.device)



    # TODO : checkpoint, history, etc
