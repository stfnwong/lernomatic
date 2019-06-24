"""
ADVERSARIAL_SEMISUPERVISED_TRAINER
Semi-Supervised Adversarial Autoencoder trainer

Stefan Wong 2019
"""

import importlib
import torch
import torch.nn.functional as F
import numpy as np
from lernomatic.models import common
from lernomatic.train import trainer

#debug
#from pudb import set_trace; set_trace()


class AAESemiTrainer(trainer.Trainer):
    def __init__(self,
                 q_net:common.LernomaticModel=None,
                 p_net:common.LernomaticModel=None,
                 d_cat_net:common.LernomaticModel=None,
                 d_gauss_net:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.q_net       = q_net
        self.p_net       = p_net
        self.d_cat_net   = d_cat_net
        self.d_gauss_net = d_gauss_net
        # keyword args specific to this trainer
        self.gen_lr    : float = kwargs.pop('gen_lr', 1e-4)
        self.reg_lr    : float = kwargs.pop('reg_lr', 5e-5)
        self.eps       : float = kwargs.pop('eps', 1e-15)
        self.data_norm : float = kwargs.pop('data_norm', 0.3081 + 0.1307)

        super(AAESemiTrainer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'AAESemiTrainer'

    def _init_optimizer(self) -> None:
        # create optimizers for each of the models
        if self.p_net is not None:
            self.p_decoder_optim = torch.optim.Adam(
                self.p_net.get_model_parameters(),
                lr = self.gen_lr
            )

        if self.q_net is not None:
            self.q_encoder_optim = torch.optim.Adam(
                self.q_net.get_model_parameters(),
                lr = self.gen_lr
            )
            self.q_generator_optim = torch.optim.Adam(
                self.q_net.get_model_parameters(),
                lr = self.reg_lr
            )

        if self.d_cat_net is not None:
            self.d_generator_optim = torch.optim.Adam(
                self.d_cat_net.get_model_parameters(),
                lr = self.reg_lr
            )

        if self.d_gauss_net is not None:
            self.d_generator_optim = torch.optim.Adam(
                self.d_gauss_net.get_model_parameters(),
                lr = self.reg_lr
            )

    def _init_history(self) -> None:
        self.loss_iter      = 0
        self.val_loss_iter  = 0
        self.acc_iter       = 0
        self.iter_per_epoch = int(len(self.train_loader) / self.num_epochs)

        self.g_loss_history       = np.zeros(len(self.train_loader) * self.num_epochs)
        self.d_cat_loss_history   = np.zeros(len(self.train_loader) * self.num_epochs)
        self.d_gauss_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)
        self.recon_loss_history   = np.zeros(len(self.train_loader) * self.num_epochs)

    def _send_to_device(self) -> None:
        if self.p_net is not None:
            self.p_net.send_to(self.device)

        if self.d_cat_net is not None:
            self.d_cat_net.send_to(self.device)

        if self.d_gauss_net is not None:
            self.d_gauss_net.send_to(self.device)

    def train_epoch(self) -> None:
        """
        TRAIN_EPOCH
        Run a single epoch of the training routine
        """

        self.p_net.set_train()
        self.q_net.set_train()
        self.d_cat_net.set_train()
        self.d_gauss_net.set_train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)


