"""
ADVERSARIAL_TRAINER
New Trainer for Adversarial Autoencoder stuff

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


class AAETrainer(trainer.Trainer):
    def __init__(self,
                 q_net:common.LernomaticModel=None,
                 p_net:common.LernomaticModel=None,
                 d_net:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.q_net = q_net
        self.p_net = p_net
        self.d_net = d_net
        # keyword args specific to this trainer
        self.gen_lr    : float = kwargs.pop('gen_lr', 1e-4)
        self.reg_lr    : float = kwargs.pop('reg_lr', 5e-5)
        self.eps       : float = kwargs.pop('eps', 1e-15)
        self.data_norm : float = kwargs.pop('data_norm', 0.3081 + 0.1307)

        super(AAETrainer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'AAETrainer'

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

        if self.d_net is not None:
            self.d_generator_optim = torch.optim.Adam(
                self.d_net.get_model_parameters(),
                lr = self.reg_lr
            )

    def _init_history(self) -> None:
        self.loss_iter      = 0
        self.val_loss_iter  = 0
        self.acc_iter       = 0

        if self.train_loader is not None:
            self.iter_per_epoch     = int(len(self.train_loader) / self.num_epochs)
            self.g_loss_history     = np.zeros(len(self.train_loader) * self.num_epochs)
            self.d_loss_history     = np.zeros(len(self.train_loader) * self.num_epochs)
            self.recon_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)
        else:
            self.iter_per_epoch     = 0
            self.g_loss_history     = None
            self.d_loss_history     = None
            self.recon_loss_history = None

    def _send_to_device(self) -> None:
        if self.p_net is not None:
            self.p_net.send_to(self.device)

        if self.q_net is not None:
            self.q_net.send_to(self.device)

        if self.d_net is not None:
            self.d_net.send_to(self.device)

    def train_epoch(self) -> None:
        """
        TRAIN_EPOCH
        Run a single epoch of the training routine
        """

        self.p_net.set_train()
        self.q_net.set_train()
        self.d_net.set_train()

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # send to device and reshape
            data = data.to(self.device)
            target = target.to(self.device)
            data.resize_(self.batch_size, self.q_net.get_x_dim())

            # init solver gradients
            self.p_decoder_optim.zero_grad()
            self.q_encoder_optim.zero_grad()
            self.q_generator_optim.zero_grad()
            self.d_generator_optim.zero_grad()

            # Reconstruction pass
            z_sample = self.q_net.forward(data)
            x_sample = self.p_net.forward(z_sample)
            recon_loss = F.binary_cross_entropy(
                x_sample + self.eps,
                data.resize(self.batch_size, self.p_net.get_x_dim()) + self.eps
            )

            recon_loss.backward()
            self.p_decoder_optim.step()
            self.q_encoder_optim.step()

            self.p_decoder_optim.zero_grad()
            self.q_encoder_optim.zero_grad()
            self.d_generator_optim.zero_grad()
            # Regularization pass

            self.q_net.set_eval()
            z_real_gauss = torch.randn(self.batch_size, self.q_net.get_z_dim()) * 5.0
            z_real_gauss = z_real_gauss.to(self.device)
            z_fake_gauss = self.q_net.forward(data)
            # reshape

            d_real_gauss = self.d_net.forward(z_real_gauss)
            d_fake_gauss = self.d_net.forward(z_fake_gauss)

            d_loss = -torch.mean(
                torch.log(d_real_gauss + self.eps) +
                torch.log(1 - d_fake_gauss + self.eps)
            )

            d_loss.backward()
            self.d_generator_optim.step()

            self.p_net.zero_grad()
            self.q_net.zero_grad()

            # Generator pass
            self.q_net.set_train()
            z_fake_gauss = self.q_net.forward(data)
            d_fake_gauss = self.d_net.forward(z_fake_gauss)
            g_loss = -torch.mean(
                torch.log(d_fake_gauss + self.eps)
            )

            g_loss.backward()
            self.q_generator_optim.step()

            # display
            if (batch_idx > 0) and (batch_idx % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         G Loss    D Loss     R Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f   %.6f   %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_loader),
                       g_loss.item(), d_loss.item(), recon_loss.item() )
                )

            # save loss history
            self.g_loss_history[self.loss_iter] = g_loss.item()
            self.d_loss_history[self.loss_iter] = d_loss.item()
            self.recon_loss_history[self.loss_iter] = recon_loss.item()
            self.loss_iter += 1

            # save checkpoint
            if self.save_every > 0 and (self.loss_iter % self.save_every) == 0:
                ck_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)
                hist_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '_history_.pkl'
                self.save_history(hist_name)


            # TODO : scheduling? Usually scheduling doesn't work well on these
            # GAN-type networks

    # Don't do anything for validation
    def val_epoch(self) -> None:
        pass

    def train(self) -> None:
        """
        TRAIN
        Since there is no notion of 'accuracy', we don't perform any
        val_epoch() routine for this trainer. Use the corresponding
        inferrer to generate new outputs from the latent space
        """
        if self.p_net is None:
            raise ValueError('No P net specified in trainer')
        if self.q_net is None:
            raise ValueError('No Q net specified in trainer')
        if self.d_net is None:
            raise ValueError('No D net specified in trainer')

        if self.save_every == -1:
            self.save_every = len(self.train_loader)

        for n in range(self.cur_epoch, self.num_epochs):
            self.train_epoch()

            # save history at the end of each epoch
            if self.save_hist:
                hist_name = self.checkpoint_dir + '/' + self.checkpoint_name + '_history.pkl'
                if self.verbose:
                    print('\t Saving history to file [%s] ' % str(hist_name))
                self.save_history(hist_name)

            self.cur_epoch += 1

    def get_trainer_params(self) -> dict:
        params = super(AAETrainer, self).get_trainer_params()
        params.update({
            'eps' : self.eps,
            'data_norm' : self.data_norm,
            'gen_lr'    : self.gen_lr,
            'reg_lr'    : self.reg_lr
        })

        return params

    # model checkpoints
    def save_checkpoint(self, fname : str) -> None:
        if self.verbose:
            print('\t Saving checkpoint (epoch %d) to [%s]' % (self.cur_epoch, fname))
        checkpoint_data = {
            # Models
            'p_net' : self.p_net.get_params() if self.p_net is not None else None,
            'q_net' : self.q_net.get_params() if self.q_net is not None else None,
            'd_net' : self.d_net.get_params() if self.d_net is not None else None,
            # Optimizers
            'p_decoder_optim'   : self.p_decoder_optim.state_dict(),
            'q_encoder_optim'   : self.q_encoder_optim.state_dict(),
            'q_generator_optim' : self.q_generator_optim.state_dict(),
            'd_generator_optim' : self.d_generator_optim.state_dict(),
            'trainer_params'    : self.get_trainer_params(),
        }
        torch.save(checkpoint_data, fname)

    def load_checkpoint(self, fname: str) -> None:
        """
        Load all data from a checkpoint
        """
        checkpoint_data = torch.load(fname)
        self.set_trainer_params(checkpoint_data['trainer_params'])

        # Load the models
        # P-Net
        model_import_path = checkpoint_data['p_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['p_net']['model_name'])
        self.p_net = mod()
        self.p_net.set_params(checkpoint_data['p_net'])
        # Q-Net
        model_import_path = checkpoint_data['q_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['q_net']['model_name'])
        self.q_net = mod()
        self.q_net.set_params(checkpoint_data['q_net'])
        # D-Net
        model_import_path = checkpoint_data['d_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['d_net']['model_name'])
        self.d_net = mod()
        self.d_net.set_params(checkpoint_data['d_net'])

        # Load the optimizers
        self._init_optimizer()

        self.p_decoder_optim.load_state_dict(checkpoint_data['p_decoder_optim'])
        for state in self.p_decoder_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.q_encoder_optim.load_state_dict(checkpoint_data['q_encoder_optim'])
        for state in self.q_encoder_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.q_generator_optim.load_state_dict(checkpoint_data['q_generator_optim'])
        for state in self.q_generator_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.d_generator_optim.load_state_dict(checkpoint_data['d_generator_optim'])
        for state in self.d_generator_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # restore trainer object info
        self._send_to_device()

    def save_history(self, filename:str) -> None:
        history = dict()
        history['loss_iter']      = self.loss_iter
        history['cur_epoch']      = self.cur_epoch
        history['iter_per_epoch'] = self.iter_per_epoch
        history['g_loss_history'] = self.g_loss_history
        history['d_loss_history'] = self.d_loss_history
        history['recon_loss_history'] = self.recon_loss_history

        torch.save(history, filename)

    def load_history(self, filename:str) -> None:
        history = torch.load(filename)
        self.loss_iter          = history['loss_iter']
        self.cur_epoch          = history['cur_epoch']
        self.iter_per_epoch     = history['iter_per_epoch']
        self.g_loss_history     = history['g_loss_history']
        self.d_loss_history     = history['d_loss_history']
        self.recon_loss_history = history['recon_loss_history']
