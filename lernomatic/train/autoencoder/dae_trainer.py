"""
DENOISING_AE_TRAINER
Trainer for a Denoising Autoencoder

Stefan Wong 2019
"""

import torch
import torch.nn as nn
import numpy as np
from lernomatic.train import trainer
from lernomatic.models import common

from pudb import set_trace; set_trace()


class DAETrainer(trainer.Trainer):
    def __init__(self,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.encoder = encoder
        self.decoder = decoder
        # TODO : options for noise?

        super(DAETrainer, self).__init__(None, **kwargs)

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

    def train_epoch(self) -> None:
        """
        Train a single epoch
        """
        self.encoder.set_train()
        self.decoder.set_train()

        for batch_idx, (data, label) in enumerate(self.train_loader):
            # prep data
            # TODO : make noise parameters settable
            noise      = torch.rand(*data.shape)
            data_noise = torch.mul(data + 0.25, 0.1 * noise)
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
        self.model = mod()
        self.model.set_params(checkpoint_data['encoder'])

        # load decoder
        model_import_path = checkpoint_data['decoder']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['decoder']['model_name'])
        self.model = mod()
        self.model.set_params(checkpoint_data['decoder'])

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
