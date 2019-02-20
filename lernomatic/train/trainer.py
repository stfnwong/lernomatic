"""
TRAINER
Module for training networks

Stefan Wong 2018
"""

import torch
from torch import nn
import numpy as np

# debug
#from pudb import set_trace; set_trace()


# TODO : layer freeze/unfreeze ?
class Trainer(object):
    """
    Trainer

    Base class for model trainers in lernomatic. Note that this is not
    an abstract class and can be instantiated.
    """
    def __init__(self, model=None, **kwargs):
        self.model           = model
        # Training loop options
        self.num_epochs      = kwargs.pop('num_epochs', 10)
        self.learning_rate   = kwargs.pop('learning_rate', 1e-4)
        self.momentum        = kwargs.pop('momentum', 0.5)
        self.weight_decay    = kwargs.pop('weight_decay', 1e-5)
        self.loss_function   = kwargs.pop('loss_function', 'CrossEntropyLoss')
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
        self.val_dataset     = kwargs.pop('val_dataset', None)
        self.shuffle         = kwargs.pop('shuffle', True)
        self.num_workers     = kwargs.pop('num_workers' , 1)
        # parameter scheduling
        self.lr_scheduler    = kwargs.pop('lr_scheduler', None)

        if self.test_batch_size == 0:
            self.test_batch_size = self.batch_size
        self.best_acc = 0.0

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

    def __str__(self):
        s = []
        s.append('Trainer :\n')
        param = self.get_trainer_params()
        for k, v in param.items():
            s.append('\t [%s] : %s\n' % (str(k), str(v)))

        return ''.join(s)

    def _init_optimizer(self):
        if self.model is not None:
            if hasattr(torch.optim, self.optim_function):
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
            loss_obj = getattr(nn, self.loss_function)
            self.criterion = loss_obj()
        else:
            raise ValueError('Cannot find loss function [%s]' % str(self.loss_function))

    def _init_history(self):
        self.loss_iter = 0
        self.test_loss_iter = 0
        self.acc_iter = 0
        self.iter_per_epoch = int(len(self.train_loader) / self.num_epochs)
        self.loss_history   = np.zeros(len(self.train_loader) * self.num_epochs)
        if self.test_loader is not None:
            self.test_loss_history = np.zeros(len(self.test_loader) * self.num_epochs)
            self.acc_history = np.zeros(len(self.test_loader) * self.num_epochs)
        else:
            self.test_loss_history = None
            self.acc_history = None

    def _init_dataloaders(self):
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
                batch_size = self.test_batch_size,
                shuffle    = self.shuffle
            )

    def _init_device(self):
        if self.device_id < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % self.device_id)

    def _send_to_device(self):
        self.model = self.model.to(self.device)

    # ======== getters, setters
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

    # common getters/setters
    def get_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
        #optim_state = self.optimizer.state_dict()
        #return optim_state['lr']

    def get_momentum(self):
        optim_state = self.optimizer.state_dict()
        if 'momentum' in optim_state:
            return optim_state['momentum']
        return None

    def set_learning_rate(self, lr, param_zero=True):
        if param_zero:
            self.optimizer.param_groups[0]['lr'] = lr
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

    def set_momentum(self, momentum):
        optim_state = self.optimizer.state_dict()
        if 'momentum' in optim_state:
            for g in self.optimizer.param_groups:
                g['momentum'] = momentum

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def get_lr_scheduler(self):
        return self.lr_scheduler

    # history getters - these provide the history up to the current iteration
    def get_loss_history(self):
        if self.loss_iter == 0:
            return None
        return self.loss_history[0 : self.loss_iter]

    def get_test_loss_history(self):
        if self.test_loss_iter == 0:
            return None
        return self.test_loss_history[0 : self.test_loss_iter]

    def get_acc_history(self):
        if self.acc_iter == 0:
            return None
        return self.acc_history[0 : self.acc_iter]

    # Basic training/test routines. Specialize these when needed
    def train_epoch(self):
        """
        TRAIN_EPOCH
        Perform training on the model for a single epoch of the dataset
        """
        self.model.train()
        # training loop
        for n, (data, target) in enumerate(self.train_loader):
            # move data
            data = data.to(self.device)
            target = target.to(self.device)

            # optimization
            output = self.model(data)
            loss   = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (n > 0) and (n % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.train_loader), loss.item()))

            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

            # save checkpoints
            if self.save_every > 0 and (self.loss_iter % self.save_every) == 0:
                ck_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)
                hist_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '_history_.pkl'
                self.save_history(hist_name)

            # perform any scheduling
            if self.lr_scheduler is not None:
                new_lr = self.lr_scheduler.get_lr(self.loss_iter)
                self.set_learning_rate(new_lr)

    def test_epoch(self):
        """
        TEST_EPOCH
        Run a single epoch on the test dataset
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0

        for n, (data, labels) in enumerate(self.test_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                output = self.model(data)
            loss = self.criterion(output, labels)
            test_loss += loss.item()

            # accuracy
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()

            if (n % self.print_every) == 0:
                print('[TEST]  :   Epoch       iteration         Test Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.test_loader), loss.item()))

            self.test_loss_history[self.test_loss_iter] = loss.item()
            self.test_loss_iter += 1

        avg_test_loss = test_loss / len(self.test_loader)
        acc = correct / len(self.test_loader.dataset)
        self.acc_history[self.acc_iter] = acc
        self.acc_iter += 1
        print('[TEST]  : Avg. Test Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
              (avg_test_loss, correct, len(self.test_loader.dataset),
               100.0 * acc)
        )

        # save the best weights
        if acc > self.best_acc:
            self.best_acc = acc
            if self.save_every > 0:
                ck_name = self.checkpoint_dir + '/' + 'best_' +  self.checkpoint_name
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)
                hist_name = self.checkpoint_dir + '/' + 'best_' + self.checkpoint_name + '_history.pkl'
                if self.verbose:
                    print('\t Saving history to file [%s] ' % str(hist_name))
                self.save_history(hist_name)

    def train(self):
        """
        TRAIN
        Standard training routine
        """
        for n in range(self.num_epochs):
            self.train_epoch()

            if self.test_loader is not None:
                self.test_epoch()
            # TODO: another validation fold?
            #if self.val_loader is not None:
            #    self.val_epoch()
            self.cur_epoch += 1
