"""
DENOISING_AE_TRAINER
Trainer for a Denoising Autoencoder

Stefan Wong 2019
"""

import importlib
import torch
import torch.nn as nn
import torchvision
import numpy as np
from lernomatic.train import trainer
from lernomatic.models import common


class DAETrainer(trainer.Trainer):
    """
    DAETrainer
    Trainer object for a denoising autoencoder.
    """
    def __init__(self,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.noise_bias:float   = kwargs.pop('noise_bias', 0.25)
        self.noise_factor:float = kwargs.pop('noise_factor', 0.1)

        super(DAETrainer, self).__init__(None, **kwargs)
        self.loss_function = 'MSELoss'      # only for printing really since we force MSELoss in _init_optimizer()

    def __repr__(self) -> str:
        return 'DAETrainer'

    def _init_optimizer(self) -> None:
        if (self.encoder is None) or (self.decoder is None):
            self.optimizer = None
        else:
            model_params = list(self.encoder.get_model_parameters()) + \
                list(self.decoder.get_model_parameters())
            if hasattr(torch.optim, self.optim_function):
                self.optimizer = getattr(torch.optim, self.optim_function)(
                    model_params,
                    lr = self.learning_rate,
                    weight_decay = self.weight_decay
                )
            else:
                raise ValueError('Cannot find optim function %s' % str(self.optim_function))

        # For this module always use MSELoss
        self.criterion = nn.MSELoss()

    def _send_to_device(self) -> None:
        if self.encoder is not None:
            self.encoder.send_to(self.device)
        if self.decoder is not None:
            self.decoder.send_to(self.device)

    def get_noise(self, X:torch.Tensor) -> torch.Tensor:
        noise = torch.rand(*X.shape)
        return torch.mul(X + self.noise_bias, self.noise_factor * noise)

    def train_epoch(self) -> None:
        """
        Train a single epoch
        """
        self.encoder.set_train()
        self.decoder.set_train()

        for batch_idx, (data, label) in enumerate(self.train_loader):
            # prep data
            data_noise = self.get_noise(data)
            data       = data.to(self.device)
            label      = label.to(self.device)
            data_noise = data_noise.to(self.device)

            self.optimizer.zero_grad()

            output = self.encoder.forward(data_noise)
            output = self.decoder.forward(output)

            loss = self.criterion(output, data)
            loss.backward()
            self.optimizer.step()

            # display
            if (batch_idx > 0) and (batch_idx % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_loader), loss.item()))

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('train/loss', loss.item(), self.loss_iter)

            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

            # save checkpoints
            if self.save_every > 0 and (self.loss_iter % self.save_every) == 0:
                ck_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_epoch_' + str(self.cur_epoch) + '_iter_' + str(self.loss_iter) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)

            # perform any scheduling
            if self.lr_scheduler is not None:
                self.apply_lr_schedule()

            if self.mtm_scheduler is not None:
                self.apply_mtm_schedule()

        # Render generated/denoised image
        if self.tb_writer is not None:
            X, _ = next(iter(self.train_loader))
            X = torch.mul(X + self.noise_bias, self.noise_factor + torch.randn(*X.shape))
            X = X.to(self.device)

            output = self.encoder.forward(X)
            output = self.decoder.forward(output)
            output = output.detach()
            outout = output.to('cpu')
            grid   = torchvision.utils.make_grid(output)
            self.tb_writer.add_image('dae/denoised', grid, self.cur_epoch)

    def save_checkpoint(self, fname:str) -> None:
        checkpoint_data = {
            'trainer_params' :  self.get_trainer_params(),
            # Models
            'encoder' : self.encoder.get_params() if self.encoder is not None else None,
            'decoder' : self.decoder.get_params() if self.decoder is not None else None,
            # Optimizers
            'optim' : self.optimizer.state_dict(),
        }
        torch.save(checkpoint_data, fname)

    def load_checkpoint(self, fname:str) -> None:
        checkpoint_data = torch.load(fname)
        self.set_trainer_params(checkpoint_data['trainer_params'])

        # load encoder
        model_import_path = checkpoint_data['encoder']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['encoder']['model_name'])
        self.encoder = mod()
        self.encoder.set_params(checkpoint_data['encoder'])

        # load decoder
        model_import_path = checkpoint_data['decoder']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['decoder']['model_name'])
        self.decoder = mod()
        self.decoder.set_params(checkpoint_data['decoder'])

        # Set up optimizer
        self._init_optimizer()
        self.optimizer.load_state_dict(checkpoint_data['optim'])
        # Transfer all the tensors to the current device
        for state in self.optimizer.state.values():
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
        history['loss_history']   = self.loss_history

        torch.save(history, filename)

    def load_history(self, filename:str) -> None:
        history = torch.load(filename)
        self.loss_iter          = history['loss_iter']
        self.cur_epoch          = history['cur_epoch']
        self.iter_per_epoch     = history['iter_per_epoch']
        self.loss_history       = history['loss_history']
