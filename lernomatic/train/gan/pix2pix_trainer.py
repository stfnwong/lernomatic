"""
PIX2PIX TRAINER
Trainer for pix2pix

Stefan Wong 2019
"""

import torch
import numpy as np
from lernomatic.models import common
from lernomatic.train import trainer
from lernomatic.models.gan import gan_loss


# debug
from pudb import set_trace; set_trace()


class Pix2PixTrainer(trainer.Trainer):
    def __init__(self,
                 d_net:common.LernomaticModel=None,
                 g_net:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.d_net            = d_net
        self.g_net            = g_net
        self.beta1     :float = kwargs.pop('beta1', 0.5)
        self.l1_lambda :float = kwargs.pop('l1_lambda', 100.0)
        self.gan_mode  :str   = kwargs.pop('gan_mode', 'vanilla')

        super(Pix2PixTrainer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'Pix2PixTrainer'

    def _send_to_device(self) -> None:
        if self.d_net is not None:
            self.d_net.send_to(self.device)
        if self.g_net is not None:
            self.g_net.send_to(self.device)

    def _init_history(self) -> None:
        self.loss_iter      = 0
        self.val_loss_iter  = 0
        self.acc_iter       = 0

        if self.train_loader is not None:
            self.iter_per_epoch = int(len(self.train_loader) / self.num_epochs)
            self.g_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)
            self.d_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)
        else:
            self.iter_per_epoch = 0
            self.g_loss_history = None
            self.d_loss_history = None

    def _init_optimizer(self) -> None:
        if self.d_net is not None:
            self.d_optim = torch.optim.Adam(
                self.d_net.get_model_parameters(),
                lr=self.learning_rate,
                betas = (self.beta1, 0.999)
            )
        else:
            self.d_optim = None

        if self.g_net is not None:
            self.g_optim = torch.optim.Adam(
                self.g_net.get_model_parameters(),
                lr=self.learning_rate,
                betas = (self.beta1, 0.999)
            )
        else:
            self.g_optim = None

        self.gan_criterion = gan_loss.GANLoss(self.gan_mode, self.device)
        self.l1_criterion  = torch.nn.L1Loss()


    def train_epoch(self) -> None:
        """
        TRAIN_EPOCH
        Train the networks on a single epoch of data
        """
        self.d_net.set_train()
        self.g_net.set_train()

        for batch_idx, (a_real, b_real) in enumerate(self.train_loader):
            a_real = a_real.to(self.device)
            b_real = b_real.to(self.device)

            # Find G(A)  (fake data)
            b_fake      = self.g_net.forward(a_real)

            # ======= Train discriminator ======== #
            self.d_optim.zero_grad()
            ab_fake     = torch.cat((a_real, b_fake), dim=1)
            pred_fake   = self.d_net.forward(ab_fake.detach()) #  remove gradient references
            # fake loss
            d_loss_fake = self.gan_criterion(pred_fake, target_real=False)
            ab_real     = torch.cat((a_real, b_real), 1)
            pred_real   = self.d_net.forward(ab_real)
            # real loss
            d_loss_real = self.gan_criterion(pred_real, target_real=True)
            d_loss      = (d_loss_fake + d_loss_real) * 0.5
            d_loss.backward()

            self.d_optim.step()

            # ======= Train generator ======== #
            self.d_net.set_eval()
            self.g_optim.zero_grad()

            pred_fake = self.d_net.forward(ab_fake)
            g_loss_gan = self.gan_criterion(pred_fake, True)
            g_loss_l1  = self.l1_criterion(b_fake, b_real) * self.l1_lambda
            g_loss     = g_loss_gan + g_loss_l1
            g_loss.backward()

            self.g_optim.step()

            # update loss history
            self.d_loss_history[self.loss_iter] = d_loss.item()
            self.g_loss_history[self.loss_iter] = g_loss.item()
            self.loss_iter += 1

            # display training progress
            if (batch_idx > 0) and (batch_idx % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         G Loss    D Loss     ')
                print('            [%3d/%3d]   [%6d/%6d]         %.6f   %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_loader),
                       g_loss.item(), d_loss.item())
                )

            # save checkpoints during training
            if self.save_every > 0 and (self.loss_iter % self.save_every) == 0:
                ck_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)
                hist_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '_history_.pkl'
                self.save_history(hist_name)


    def val_epoch(self) -> None:
        # No validation for GAN
        pass

    def save_checkpoint(self, fname:str) -> None:
        checkpoint_data = {
            # models
            'd_net'   : self.d_net.get_params(),
            'g_net'   : self.g_net.get_params(),
            # optimizers
            'd_optim' : self.d_optim.state_dict(),
            'g_optim' : self.g_optim.state_dict(),
            'trainer_params' : self.get_trainer_params()
        }
        torch.save(checkpoint_data,fname)


    def load_checkpoint(self, fname:str) -> None:
        checkpoint_data = torch.load(fname)
        # get generator
        model_import_path = checkpoint_data['g_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['g_net']['model_name'])
        self.g_net = mod()
        self.g_net.set_params(checkpoint_data['g_net'])
        # get discriminator
        model_import_path = checkpoint_data['d_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['d_net']['model_name'])
        self.d_net = mod()
        self.d_net.set_params(checkpoint_data['d_net'])

        # load generator optimizer
        self._init_optimizer()
        self.g_optim.load_state_dict(checkpoint_data['g_optim'])
        # Transfer all the tensors to the current device
        for state in self.g_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # load discriminator optimizer
        self.g_optim.load_state_dict(checkpoint_data['g_optim'])
        # Transfer all the tensors to the current device
        for state in self.g_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # restore trainer object info
        self._send_to_device()

    def save_history(self, fname:str) -> None:
        history = dict()
        history['loss_iter'] = self.loss_iter
        history['val_loss_iter']  = self.val_loss_iter
        history['acc_iter']       = self.acc_iter

        if self.train_loader is not None:
            history['iter_per_epoch'] = self.iter_per_epoch
            history['g_loss_history'] = self.g_loss_history
            history['d_loss_history'] = self.d_loss_history

        torch.save(history, fname)

    def load_history(self, fname:str) -> None:
        history = torch.load(fname)

        self.loss_iter = history['loss_iter']
        self.val_loss_iter = history['val_loss_iter']
        self.acc_iter = history['acc_iter']
        self.g_loss_history = history['g_loss_history']
        self.d_loss_history = history['d_loss_history']
        self.iter_per_epoch = history['iter_per_epoch']
