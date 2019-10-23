"""
GRID_SEARCH
Tools to perform grid search of hyperparameters

Stefan Wong 2019
"""

import torch
import numpy as np

from lernomatic.models import common
from lernomatic.train import trainer


class GridResult(object):
    def __init__(self, param:dict=None, acc:float=0.0) -> None:
        self.param:dict              = param
        self.acc:float               = acc
        # loss and acc history...
        self.acc_history:np.ndarray  = None
        self.loss_history:np.ndarray = None
        self.trainer_params:dict     = dict()

    def __repr__(self) -> str:
        return 'GridResult'

    def __str__(self) -> str:
        s = []
        s.append('%s\n' % repr(self))
        for k, v in self.param.items():
            s.append('\t[%s] : %s\n' % (str(k), str(v)))

        return ''.join(s)


class GridSearcher(object):
    def __init__(self,
                 model:common.LernomaticModel=None,
                 tr:trainer.Trainer=None,
                 **kwargs) -> None:
        valid_params = ('learning_rate', 'weight_decay', 'dropout', 'momentum')
        self.model   = model
        self.trainer = tr
        self.params:dict     = kwargs.pop('params', None)
        self.dont_train:bool = kwargs.pop('dont_train', False)   # really only useful for testing
        self.verbose:bool    = kwargs.pop('verbose', False)

        # history
        self.param_history = []

        # how many params to search over
        self.num_params:int = kwargs.pop('num_params', 8)
        # cycle parameters
        self.max_num_epochs:int = kwargs.pop('max_num_epochs', 10)
        self.min_num_epochs:int = kwargs.pop('min_num_epochs', 4)
        self.min_req_acc:float  = kwargs.pop('min_req_acc', 0.4)    # reject acc less than this after min_num_epochs

        self.original_trainer_params = dict()

    def __repr__(self) -> str:
        return 'GridSearcher'

    #def _init_history(self) -> None:
    #    pass

    def get_best_params(self) -> GridResult:
        best_acc_idx = 0
        best_acc = 0.0

        for idx, result in enumerate(self.param_history):
            if result.acc > best_acc:
                best_acc_idx = n
                best_acc = result.acc

        return self.param_history[best_acc_idx]

    def search(self, params:dict) -> None:
        # I seem to remember something about using logarithmic params
        if 'learning_rate' in params:
            lr_range = np.linspace(
                params['learning_rate'][0],
                params['learning_rate'][1],
                self.num_params
            )
        else:
            lr_range = [0.0]

        if 'weight_decay' in params:
            wd_range = np.linspace(
                params['weight_decay'][0],
                params['weight_decay'][1],
                self.num_params
            )
        else:
            wd_range = [0.0]

        if 'dropout' in params:
            dp_range = np.linspace(
                params['dropout'][0],
                params['dropout'][1],
                self.num_params
            )
        else:
            dp_range = [0.0]

        if 'momentum' in params:
            mn_range = np.linspace(
                params['momentum'][0],
                params['momentum'][1],
                self.num_params
            )
        else:
            mn_range = [0.0]

        self.original_trainer_params = self.trainer.get_trainer_params()

        # Do the actual grid search
        total_params = len(lr_range) + len(wd_range) + len(dp_range) + len(mn_range)
        cur_param = 0
        for lr in lr_range:
            for wd in wd_range:
                for dp in dp_range:
                    for mn in mn_range:
                        print('Searching param combination [%d / %d]' % (cur_param, total_params))

                        if lr != 0.0:
                            self.trainer.set_learning_rate(lr)
                        if wd != 0.0:
                            self.trainer.set_weight_decay(wd)
                        if dp != 0.0:
                            self.trainer.set_dropout(dp)
                        if mn != 0.0:
                            self.trainer.set_momentum(mn)

                        if not self.dont_train:
                            self.trainer.restart_history()
                            self.trainer.set_num_epochs(self.max_num_epochs)
                            for epoch in range(self.max_num_epochs):
                                #self.trainer.train()
                                self.trainer.train_epoch()
                                self.trainer.val_epoch()

                                cur_acc = self.trainer.get_cur_acc()
                                if (epoch >= self.min_num_epochs) and (cur_acc > self.min_req_acc):
                                    grid_result = GridResult()
                                    grid_result.loss_history   = self.trainer.get_loss_history()
                                    grid_result.acc_histoy     = self.trainer.get_acc_history()
                                    grid_result.trainer_params = self.trainer.get_trainer_params()
                                    grid_result.acc            = self.trainer.get_best_acc()
                                    # Copy the params that produced the result
                                    grid_params = {k: 0 for k in params.keys()}
                                    grid_params['learning_rate'] = lr
                                    grid_params['weight_decay']  = wd
                                    grid_params['dropout']       = dp
                                    grid_params['momentum']      = mn
                                    grid_result.params = grid_params
                                    self.param_history.append(grid_result)

                                    if self.verbose:
                                        print('Found grid result :\n %s' % str(grid_result))

                                    break

                                self.trainer.cur_epoch += 1

                        cur_param += 1

        # reset the trainer params
        self.trainer.set_trainer_params(self.original_trainer_params)
