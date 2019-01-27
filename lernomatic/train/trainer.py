"""
TRAINER
Module for training networks

Stefan Wong 2018
"""

import torch
from torch import nn
import numpy as np

#from lernomatic.param import learning_rate

# debug
from pudb import set_trace; set_trace()


# Other trainers should inherit from this...
class Trainer(object):
    def __init__(self, model=None, **kwargs):
        self.model           = model
        # Training loop options
        self.num_epochs      = kwargs.pop('num_epochs', 10)
        self.learning_rate   = kwargs.pop('learning_rate', 1e-4)
        self.momentum        = kwargs.pop('momentum', 0.5)
        self.weight_decay    = kwargs.pop('weight_decay', 1e-5)
        self.loss_function   = kwargs.pop('loss_function', 'BCELoss')
        self.optim_function  = kwargs.pop('optim_function', 'Adam')
        self.cur_epoch       = 0
        # validation options
        # checkpoint options
        self.checkpoint_dir  = kwargs.pop('checkpoint_dir', 'checkpoint')
        self.checkpoint_name = kwargs.pop('checkpoint_name', 'ck')
        # Internal options
        self.verbose         = kwargs.pop('verbose', True)
        self.print_every     = kwargs.pop('print_every', 10)
        self.save_every      = kwargs.pop('save_every', 1000)  # unit is iterations
        # Device options
        self.device_id       = kwargs.pop('device_id', -1)
        # dataset/loader options
        self.batch_size      = kwargs.pop('batch_size', 64)
        self.test_batch_size = kwargs.pop('test_batch_size', 0)
        self.train_dataset   = kwargs.pop('train_dataset', None)
        self.test_dataset    = kwargs.pop('test_dataset', None)
        self.shuffle         = kwargs.pop('shuffle', True)
        self.num_workers     = kwargs.pop('num_workers' , 1)

        if self.test_batch_size == 0:
            self.test_batch_size = self.batch_size

        # Setup optimizer. If we have no model then assume it will be
        self._init_optimizer()
        # set up device
        self._init_device()
        # Init the internal dataloader options. If nothing provided assume that
        # we will load options in later (eg: from checkpoint)
        self._init_dataloaders()
        # Init the loss and accuracy history. If no train_loader is provided
        # then we assume that one will be loaded later (eg: in some checkpoint
        # data)
        if self.train_loader is not None:
            self._init_history()

        self._send_to_device()

    def __repr__(self):
        return 'Trainer (%d epochs)' % self.num_epochs

    def _init_optimizer(self):
        if self.model is not None:
            if hasattr(torch.optim, self.optim_function):
                #self.optimizer = torch.optim.Adam(
                self.optimizer = getattr(torch.optim, self.optim_function)(
                    self.model.parameters(),
                    lr = self.learning_rate,
                    weight_decay = self.weight_decay
                )
            else:
                raise ValueError('Cannot find optim function %s' % str(self.optim_function))
        else:
            self.optimizer = None

        # Get a loss function
        if hasattr(nn, self.loss_function):
            self.criterion = nn.BCELoss()
        else:
            raise ValueError('Cannot find loss function [%s]' % str(self.loss_function))

    def _init_history(self):
        # TODO : add accuracy here, update trainer to perform validation on val
        # dataset (need another loader for this)
        self.loss_iter = 0
        self.acc_iter = 0
        self.iter_per_epoch = int(len(self.train_loader) / self.num_epochs)
        self.loss_history   = np.zeros(len(self.train_loader) * self.num_epochs)
        if self.test_loader is not None:
            self.test_loss_history = np.zeros(len(self.test_loader))
        else:
            self.test_loss_history = None

    def _init_dataloaders(self):
        # TODO; may want to re-use this dataset prototype elsewhere..
        #train_dataset = data.AvetronDataset(
        #    self.train_data_path,
        #    num_workers = self.num_workers,
        #    verbose = self.verbose
        #)

        if self.train_dataset is None:
            self.train_loader = None
        else:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                shuffle = self.shuffle
            )

        if self.test_dataset is None:
            self.test_loader = None
        else:
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size = self.batch_size,
                shuffle    = self.shuffle
            )

    def _init_device(self):
        if self.device_id < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % self.device_id)

    def _send_to_device(self):
        self.model = self.model.to(self.device)

    # default param options
    def get_trainer_params(self):
        params = dict()
        params['num_epochs']      = self.num_epochs
        params['learning_rate']   = self.learning_rate
        params['momentum']        = self.momentum
        params['weight_decay']    = self.weight_decay
        params['loss_function']   = self.loss_function
        params['optim_function']  = self.optim_function
        params['cur_epoch']       = self.cur_epoch
        params['iter_per_epoch']  = self.iter_per_epoch
        params['device_id']       = self.device_id
        # also get print, save params
        params['save_every']      = self.save_every
        params['print_every']     = self.print_every
        # dataloader params (to regenerate data loader)
        params['batch_size']      = self.batch_size
        params['shuffle']         = self.shuffle

        return params

    def set_trainer_params(self, params):
        self.num_epochs      = params['num_epochs']
        self.learning_rate   = params['learning_rate']
        self.momentum        = params['momentum']
        self.weigh_decay     = params['weight_decay']
        self.loss_function   = params['loss_function']
        self.optim_function  = params['optim_function']
        self.cur_epoch       = params['cur_epoch']
        self.iter_per_epoch  = params['iter_per_epoch']
        self.save_every      = params['save_every']
        self.print_every     = params['print_every']
        self.device_id       = params['device_id']
        # dataloader params
        self.batch_size      = params['batch_size']
        self.shuffle         = params['shuffle']

        self._init_device()
        self._init_dataloaders()

    def get_model_params(self):
        if self.model is None:
            return None
        return self.model.state_dict()

    # Default save and load methods
    def save_state(self, fname):
        raise NotImplementedError('This method should be implemented in the derived class')

    def load_state(self, fname):
        raise NotImplementedError('This method should be implemented in the derived class')

    # Basic training/test routines. Specialize these when needed
    def train_step(self, data, target):
        data = data.to(self.device)
        target = target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss   = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def train_fixed(self, num_iter):
        self.model.train()
        # training loop
        for n, (data, target) in enumerate(self.train_loader):
            # move data
            data = data.to(self.device)
            target = target.to(self.device)

            # optimization
            self.optimizer.zero_grad()
            output = self.model(data)
            loss   = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if (n % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.train_loader), loss.item()))

            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

            if n >= num_iter:
                return
