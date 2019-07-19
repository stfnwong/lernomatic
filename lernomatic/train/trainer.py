"""
TRAINER
Module for training networks

Stefan Wong 2018
"""

import importlib
import torch
from torch import nn
import numpy as np
from lernomatic.train import schedule
from lernomatic.models import common

# debug
#from pudb import set_trace; set_trace()


class Trainer(object):
    """
    Trainer

    Base class for model trainers in lernomatic. Note that this is not
    an abstract class and can be instantiated.
    """
    def __init__(self, model:common.LernomaticModel=None, **kwargs) -> None:
        self.model           = model
        # Training loop options
        self.num_epochs      :int   = kwargs.pop('num_epochs', 10)
        self.learning_rate   :float = kwargs.pop('learning_rate', 1e-4)
        self.momentum        :float = kwargs.pop('momentum', 0.5)
        self.weight_decay    :float = kwargs.pop('weight_decay', 1e-5)
        self.loss_function   :str   = kwargs.pop('loss_function', 'CrossEntropyLoss')
        self.optim_function  :str   = kwargs.pop('optim_function', 'Adam')
        self.cur_epoch       :int   = 0
        # validation options
        # checkpoint options
        self.checkpoint_dir  :str   = kwargs.pop('checkpoint_dir', 'checkpoint')
        self.checkpoint_name :str   = kwargs.pop('checkpoint_name', 'ck')
        self.save_hist       :bool  = kwargs.pop('save_hist', True)
        # Internal options
        self.verbose         :float = kwargs.pop('verbose', True)
        self.print_every     :int   = kwargs.pop('print_every', 10)
        self.save_every      :float = kwargs.pop('save_every', -1)  # unit is iterations, -1 = save every epoch
        self.save_best       :float = kwargs.pop('save_best', False)
        # Device options
        self.device_id       :int   = kwargs.pop('device_id', -1)
        self.device_map      :float = kwargs.pop('device_map', None)
        # dataset/loader options
        self.batch_size      :int   = kwargs.pop('batch_size', 64)
        self.val_batch_size  :int   = kwargs.pop('val_batch_size', 0)
        self.train_dataset          = kwargs.pop('train_dataset', None)
        self.test_dataset           = kwargs.pop('test_dataset', None)
        self.val_dataset            = kwargs.pop('val_dataset', None)
        self.shuffle         :float = kwargs.pop('shuffle', True)
        self.num_workers     :int   = kwargs.pop('num_workers' , 1)
        self.drop_last       :bool  = kwargs.pop('drop_last', True)
        # parameter scheduling
        self.lr_scheduler           = kwargs.pop('lr_scheduler', None)
        self.mtm_scheduler          = kwargs.pop('mtm_scheduler', None)
        self.stop_when_acc   :float = kwargs.pop('stop_when_acc', 0.0)
        self.early_stop      :dict  = kwargs.pop('early_stop', None)

        if self.val_batch_size == 0:
            self.val_batch_size = self.batch_size
        self.best_acc = 0.0
        if self.save_every > 0:
            self.save_best = True

        # set up device
        self._init_device()
        # Setup optimizer. If we have no model then assume it will be
        self._init_optimizer()
        # Init the internal dataloader options. If nothing provided assume that
        # we will load options in later (eg: from checkpoint)
        self._init_dataloaders()
        # Init the loss and accuracy history. If no train_loader is provided
        # then we assume that one will be loaded later (eg: in some checkpoint
        # data)
        self._init_history()

        self._send_to_device()

    def __repr__(self) -> str:
        return 'Trainer (%d epochs)' % self.num_epochs

    def __str__(self) -> str:
        s = []
        s.append('Trainer :\n')
        param = self.get_trainer_params()
        for k, v in param.items():
            s.append('\t [%s] : %s\n' % (str(k), str(v)))

        return ''.join(s)

    def _init_optimizer(self) -> None:
        if self.model is not None:
            if hasattr(torch.optim, self.optim_function):
                self.optimizer = getattr(torch.optim, self.optim_function)(
                    self.model.get_model_parameters(),
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

    def _init_history(self) -> None:
        self.loss_iter      = 0
        self.val_loss_iter  = 0
        self.acc_iter       = 0
        self.iter_per_epoch = int(len(self.train_loader) / self.num_epochs)

        if self.train_loader is not None:
            self.loss_history   = np.zeros(len(self.train_loader) * self.num_epochs)
        else:
            self.loss_history = None

        if self.val_loader is not None:
            self.val_loss_history = np.zeros(len(self.val_loader) * self.num_epochs)
            self.acc_history = np.zeros(len(self.val_loader) * self.num_epochs)
        else:
            self.val_loss_history = None
            self.acc_history = None

    def _init_dataloaders(self) -> None:
        if self.train_dataset is None:
            self.train_loader = None
        else:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                drop_last = self.drop_last,
                shuffle = self.shuffle
            )

        if self.test_dataset is None:
            self.test_loader = None
        else:
            self.test_loader = torch.utils.data.DataLoader(
                self.test_dataset,
                batch_size = self.val_batch_size,
                drop_last = self.drop_last,
                shuffle    = self.shuffle
            )

        if self.val_dataset is None:
            self.val_loader = None
        else:
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size = self.val_batch_size,
                drop_last = self.drop_last,
                shuffle    = False
            )

    def _init_device(self) -> None:
        if self.device_id < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % self.device_id)

    def _send_to_device(self) -> None:
        self.model.send_to(self.device)

    def set_num_epochs(self, num_epochs:int) -> None:
        if num_epochs > self.num_epochs:
            # resize history
            temp_loss_history = np.copy(self.loss_history)
            if self.val_loss_history is not None:
                temp_val_loss_history = np.copy(self.val_loss_history)
            if self.acc_history is not None:
                temp_acc_history = np.copy(self.acc_history)
            temp_loss_iter = self.loss_iter
            temp_val_loss_iter = self.val_loss_iter
            temp_acc_iter = self.acc_iter
            self.num_epochs = num_epochs
            self._init_history()
            # restore old history
            self.loss_history[:len(temp_loss_history)] = temp_loss_history
            if self.val_loss_history is not None:
                self.val_loss_history[:len(temp_val_loss_history)] = temp_val_loss_history
            if self.acc_history is not None:
                self.acc_history[:len(temp_acc_history)] = temp_acc_history
            self.loss_iter = temp_loss_iter
            self.val_loss_iter = temp_val_loss_iter
            self.acc_iter = temp_acc_iter
        else:
            self.num_epochs = num_epochs

    def get_num_epochs(self) -> int:
        return self.num_epochs

    def get_cur_epoch(self) -> int:
        return self.cur_epoch

    # ======== getters, setters
    def get_model(self) -> common.LernomaticModel:
        return self.model

    def get_model_params(self) -> dict:
        if self.model is None:
            return None
        return self.model.get_net_state_dict()

    # common getters/setters
    def get_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr: float, param_zero:bool=True) -> None:
        if param_zero:
            self.optimizer.param_groups[0]['lr'] = lr
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

    def get_momentum(self) -> float:
        optim_state = self.optimizer.state_dict()
        if 'momentum' in optim_state:
            return optim_state['momentum']
        return None

    def set_momentum(self, momentum: float) -> None:
        optim_state = self.optimizer.state_dict()
        if 'momentum' in optim_state:
            for g in self.optimizer.param_groups:
                g['momentum'] = momentum

    def set_lr_scheduler(self, lr_scheduler: schedule.LRScheduler) -> None:
        self.lr_scheduler = lr_scheduler

    def get_lr_scheduler(self) -> schedule.LRScheduler:
        return self.lr_scheduler

    def set_mtm_scheduler(self, mtm_scheduler) -> None:
        self.mtm_scheduler = mtm_scheduler

    def get_mtm_scheduler(self) -> schedule.LRScheduler:
        return self.mtm_scheduler

    def apply_lr_schedule(self) -> None:
        if isinstance(self.lr_scheduler, schedule.TriangularDecayWhenAcc):
            new_lr = self.lr_scheduler.get_lr(self.loss_iter, self.acc_history[self.acc_iter])
        elif isinstance(self.lr_scheduler, schedule.EpochSetScheduler) or isinstance(self.lr_scheduler, schedule.DecayWhenEpoch):
            new_lr = self.lr_scheduler.get_lr(self.cur_epoch)
        elif isinstance(self.lr_scheduler, schedule.DecayWhenAcc):
            new_lr = self.lr_scheduler.get_lr(self.acc_history[self.acc_iter])
        else:
            new_lr = self.lr_scheduler.get_lr(self.loss_iter)
        self.set_learning_rate(new_lr)

    def apply_mtm_schedule(self) -> None:
        if isinstance(self.mtm_scheduler, schedule.TriangularDecayWhenAcc):
            new_mtm = self.mtm_scheduler.get_lr(self.loss_iter, self.acc_history[self.acc_iter])
        elif isinstance(self.mtm_scheduler, schedule.EpochSetScheduler) or isinstance(self.mtm_scheduler, schedule.DecayWhenEpoch):
            new_mtm = self.mtm_scheduler.get_lr(self.cur_epoch)
        elif isinstance(self.mtm_scheduler, schedule.DecayWhenAcc):
            new_mtm = self.mtm_scheduler.get_lr(self.acc_history[self.acc_iter])
        else:
            new_mtm = self.mtm_scheduler.get_lr(self.loss_iter)
        self.set_momentum(new_mtm)

    # Layer freeze / unfreeze
    def freeze_to(self, layer_num: int) -> None:
        """
        Freeze layers in model from the start of the network forwards
        """
        for n, param in enumerate(self.model.parameters()):
            param.requires_grad = False
            if n >= layer_num:
                break

    def unfreeze_to(self, layer_num: int) -> None:
        """
        Unfreeze layers in model from the start of the network forwards
        """
        for n, param in enumerate(self.model.parameters()):
            param.requires_grad = True
            if n >= layer_num:
                break

    # Basic training/test routines. Specialize these when needed
    def train_epoch(self) -> None:
        """
        TRAIN_EPOCH
        Perform training on the model for a single epoch of the dataset
        """
        self.model.set_train()
        # training loop
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # move data
            data = data.to(self.device)
            target = target.to(self.device)

            # optimization
            output = self.model.forward(data)
            loss   = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (batch_idx > 0) and (batch_idx % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_loader), loss.item()))

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
                self.apply_lr_schedule()

            if self.mtm_scheduler is not None:
                self.apply_mtm_schedule()

    def val_epoch(self) -> None:
        """
        VAL_EPOCH
        Run a single epoch on the test dataset
        """
        self.model.set_eval()
        val_loss = 0.0
        correct = 0

        for batch_idx, (data, labels) in enumerate(self.val_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                output = self.model.forward(data)
            loss = self.criterion(output, labels)
            val_loss += loss.item()

            # accuracy
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()

            if (batch_idx % self.print_every) == 0:
                print('[VAL ]  :   Epoch       iteration         Test Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.val_loader), loss.item()))

            self.val_loss_history[self.val_loss_iter] = loss.item()
            self.val_loss_iter += 1

        avg_val_loss = val_loss / len(self.val_loader)
        acc = correct / len(self.val_loader.dataset)
        self.acc_history[self.acc_iter] = acc
        self.acc_iter += 1
        print('[VAL ]  : Avg. Test Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
              (avg_val_loss, correct, len(self.val_loader.dataset),
               100.0 * acc)
        )

        # save the best weights
        if acc > self.best_acc:
            self.best_acc = acc
            if self.save_best is True:
                ck_name = self.checkpoint_dir + '/' + 'best_' +  self.checkpoint_name + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)

    def train(self) -> None:
        """
        TRAIN
        Standard training routine
        """
        if self.save_every == -1:
            self.save_every = len(self.train_loader)

        for epoch in range(self.cur_epoch, self.num_epochs):
            self.train_epoch()

            if self.val_loader is not None:
                self.val_epoch()

            # save history at the end of each epoch
            if self.save_hist:
                hist_name = self.checkpoint_dir + '/' + self.checkpoint_name + '_history.pkl'
                if self.verbose:
                    print('\t Saving history to file [%s] ' % str(hist_name))
                self.save_history(hist_name)

            # check we have reached the required accuracy and can stop early
            if self.stop_when_acc > 0.0 and self.val_loader is not None:
                if self.acc_history[self.acc_iter] >= self.stop_when_acc:
                    return

            # check if we need to perform early stopping
            if self.early_stop is not None:
                if self.cur_epoch > self.early_stop['num_epochs']:
                    acc_then = self.acc_history[self.acc_iter - self.early_stop['num_epochs']]
                    acc_now  = self.acc_history[self.acc_iter]
                    acc_delta = acc_now - acc_then
                    if acc_delta < self.early_stop['improv']:
                        if self.verbose:
                            print('[%s] Stopping early at epoch %d' % (repr(self), self.cur_epoch))
                        return

            self.cur_epoch += 1

    # history getters - these provide the history up to the current iteration
    def get_loss_history(self) -> np.ndarray:
        if self.loss_iter == 0:
            return None
        return self.loss_history[0 : self.loss_iter]

    def get_val_loss_history(self) -> np.ndarray:
        if self.val_loss_iter == 0:
            return None
        return self.val_loss_history[0 : self.val_loss_iter]

    def get_acc_history(self) -> np.ndarray:
        if self.acc_iter == 0:
            return None
        return self.acc_history[0 : self.acc_iter]

    # model checkpoints
    def save_checkpoint(self, fname : str) -> None:
        if self.verbose:
            print('\t Saving checkpoint (epoch %d) to [%s]' % (self.cur_epoch, fname))
        checkpoint_data = {
            'model' : self.model.get_params(),
            'optim' : self.optimizer.state_dict(),
            'trainer_params' : self.get_trainer_params(),
        }
        torch.save(checkpoint_data, fname)

    def load_checkpoint(self, fname: str) -> None:
        """
        Load all data from a checkpoint
        """
        checkpoint_data = torch.load(fname)
        self.set_trainer_params(checkpoint_data['trainer_params'])
        # here we just load the object that derives from LernomaticModel. That
        # object will in turn load the actual nn.Module data from the
        # checkpoint data with the 'model' key
        model_import_path = checkpoint_data['model']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['model']['model_name'])
        self.model = mod()
        self.model.set_params(checkpoint_data['model'])

        # Load optimizer
        self._init_optimizer()
        self.optimizer.load_state_dict(checkpoint_data['optim'])
        # Transfer all the tensors to the current device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # restore trainer object info
        self._send_to_device()

    def load_model_checkpoint(self, fname:str) -> None:
        """
        Load only the model component of a checkpoint. Trainer parameters
        are not affected
        """
        checkpoint_data = torch.load(fname)
        model_import_path = checkpoint_data['model']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['model']['model_name'])
        self.model = mod()
        self.model.set_params(checkpoint_data['model'])
        self._send_to_device()

    # Trainer parameters
    def get_trainer_params(self) -> dict:
        params = dict()
        params['num_epochs']      = self.num_epochs
        params['learning_rate']   = self.learning_rate
        params['momentum']        = self.momentum
        params['weight_decay']    = self.weight_decay
        params['loss_function']   = self.loss_function
        params['optim_function']  = self.optim_function
        params['cur_epoch']       = self.cur_epoch
        params['iter_per_epoch']  = self.iter_per_epoch
        # also get print, save params
        params['save_every']      = self.save_every
        params['print_every']     = self.print_every
        # dataloader params (to regenerate data loader)
        params['batch_size']      = self.batch_size
        params['val_batch_size']  = self.val_batch_size
        params['shuffle']         = self.shuffle

        return params

    def set_trainer_params(self, params: dict) -> None:
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
        # dataloader params
        self.batch_size      = params['batch_size']
        self.val_batch_size = params['val_batch_size']
        self.shuffle         = params['shuffle']

        self._init_device()
        self._init_dataloaders()
