"""
LRGANTRAINER
Trainer module for LRGAN experiments.
See LR GAN paper (https://arxiv.org/pdf/1703.01560.pdf)

Stefan Wong 2019
"""

import torch
import torch.nn as nn
import importlib
import numpy as np

from lernomatic.train import trainer
from lernomatic.models import common
from lernomatic.models.gan import lrgan
from lernomatic.util.gan import gan_util
# type stuff
from typing import Tuple

# debug
#from pudb import set_trace; set_trace()


class LRGANTrainer(trainer.Trainer):
    def __init__(self,
                 D:common.LernomaticModel=None,
                 G:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.discriminator :common.LernomaticModel = D
        self.generator     :common.LernomaticModel = G
        self.beta1         :float = kwargs.pop('beta1', 0.5)
        self.real_label    :int   = kwargs.pop('real_label', 1)
        self.fake_label    :int   = kwargs.pop('fake_label', 0)

        super(LRGANTrainer, self).__init__(None, **kwargs)

        self.loss_function = 'BCELoss'
        self.optim_function = 'Adam'
        self.criterion = nn.BCELoss()

    def __repr__(self) -> str:
        return 'LRGANTrainer'

    def _init_dataloaders(self) -> None:
        if self.train_dataset is None:
            self.train_loader = None
        else:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                shuffle = self.shuffle,
                drop_last = True
            )
        self.test_loader = None
        self.val_loader = None

    def _init_history(self) -> None:
        self.loss_iter = 0
        self.test_loss_iter = 0
        self.acc_iter = 0
        self.iter_per_epoch = int(len(self.train_loader) / self.num_epochs)
        self.d_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)
        self.g_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)

    def _init_optimizer(self) -> None:
        # generator
        if self.generator is None:
            self.optim_g = None
        else:
            self.optim_g = torch.optim.Adam(
                self.generator.get_model_parameters(),
                lr = self.learning_rate,
                betas = (self.beta1, 0.999)
            )
        # discriminator
        if self.discriminator is None:
            self.optim_d = None
        else:
            self.optim_d = torch.optim.Adam(
                self.discriminator.get_model_parameters(),
                lr = self.learning_rate,
                betas = (self.beta1, 0.999)
            )

    def _send_to_device(self) -> None:
        if self.generator is not None:
            self.generator.send_to(self.device)
        if self.discriminator is not None:
            self.discriminator.send_to(self.device)

    def train_epoch(self) -> None:
        pass

    def val_epoch(self) -> None:
        pass

    # also need to overload some of the history functions
    def get_loss_history(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.g_loss_history[0 : self.loss_iter], self.d_loss_history[0 : self.loss_iter])

    def get_g_loss_history(self) -> np.ndarray:
        return self.g_loss_history[0 : self.loss_iter]

    def get_d_loss_history(self) -> np.ndarray:
        return self.d_loss_history[0 : self.loss_iter]

    def get_test_loss_history(self) -> None:
        return None

    # Checkpointing
    def save_checkpoint(self, fname:str) -> None:
        checkpoint = {
            'trainer_params' : self.get_trainer_params(),
            'discriminator'  : self.discriminator.get_params(),
            'generator'      : self.generator.get_params(),
            'optim_d'        : self.optim_d.state_dict(),
            'optim_g'        : self.optim_g.state_dict()
        }
        torch.save(checkpoint, fname)

    def load_checkpoint(self, fname: str) -> None:
        checkpoint_data = torch.load(fname)
        # here we just load the object that derives from LernomaticModel. That
        # object will in turn load the actual nn.Module data from the
        # checkpoint data with the 'model' key

        # load generator
        gen_import_path = checkpoint_data['generator']['model_import_path']
        gen_imp = importlib.import_module(gen_import_path)
        gen = getattr(gen_imp, checkpoint_data['generator']['model_name'])
        self.generator = gen()
        self.generator.set_params(checkpoint_data['generator'])

        # load discriminator
        dis_import_path = checkpoint_data['discriminator']['model_import_path']
        dis_imp = importlib.import_module(gen_import_path)
        dis = getattr(dis_imp, checkpoint_data['discriminator']['model_name'])
        self.discriminator = dis()
        self.discriminator.set_params(checkpoint_data['discriminator'])

        # Load optimizer
        self._init_optimizer()
        self.optim_d.load_state_dict(checkpoint_data['optim_d'])
        self.optim_g.load_state_dict(checkpoint_data['optim_g'])
        # transfer tensors to current device
        for state in self.optim_d.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        for state in self.optim_g.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # restore trainer object info
        self.set_trainer_params(checkpoint_data['trainer_params'])
        self._send_to_device()

    # History
    def save_history(self, fname:str) -> None:
        history = {
            'd_loss_history' : self.d_loss_history,
            'g_loss_history' : self.g_loss_history,
            'loss_iter'      : self.loss_iter,
            'iter_per_epoch' : self.iter_per_epoch,
            'cur_epoch'      : self.cur_epoch
        }
        torch.save(history, fname)

    def load_history(self, fname:str) -> None:
        history = torch.load(fname)
        self.d_loss_history = history['d_loss_history']
        self.g_loss_history = history['g_loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch      = history['cur_epoch']
