"""
PARAM_SEARCH
Tools to search for hyperparameters

Stefan Wong 2019
"""

import pickle
import torch
import numpy as np

from lernomatic.models import common
from lernomatic.train import trainer


class SearchResult(object):
    """
    SearchResult.
    Namspace class to hold information about a partocular iteration of search.
    """
    def __init__(self, param:dict=None, acc:float=0.0) -> None:
        self.params:dict              = param
        self.acc:float               = acc
        # loss and acc history...
        self.acc_history:np.ndarray  = None
        self.loss_history:np.ndarray = None
        self.trainer_params:dict     = dict()

    def __repr__(self) -> str:
        return 'SearchResult'

    def __str__(self) -> str:
        s = []
        s.append('%s\n' % repr(self))
        for k, v in self.params.items():
            s.append('\t[%s] : %s\n' % (str(k), str(v)))
        s.append('acc : %f\n' % self.acc)
        s.append('num epochs : %d\n' % len(self.acc_history))

        return ''.join(s)

    def __getstate__(self) -> dict:
        state = dict()
        state['trainer_params'] = self.trainer_params
        state['loss_history']   = self.loss_history
        state['acc_history']    = self.acc_history
        state['acc']            = self.acc
        state['params']          = self.params

    def __setstate__(self, state:dict) -> None:
        self.trainer_params = state['trainer_params']
        self.loss_history   = state['loss_history']
        self.acc_history    = state['acc_history']
        self.acc            = state['acc']
        self.params         = state['params']



class GridSearcher(object):
    """
    GridSearcher
    Perform grid search over a set of parameters

    """
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
        self.total_num_params:int = 0

        self.original_trainer_params = dict()

    def __repr__(self) -> str:
        return 'GridSearcher'

    def _init_history(self) -> None:
        self.param_history = []

    def save_history(self, prefix:str=None) -> None:
        for n, result in enumerate(self.param_history):
            if prefix is not None:
                fname = prefix + '_search_result_' + str(n) + '.pkl'
            else:
                fname = 'search_result' + str(n) + '.pkl'
            with open(fname, 'w') as fp:
                pickle.dumps(result)

    def get_best_params(self) -> SearchResult:
        best_acc_idx = 0
        best_acc = 0.0

        for idx, result in enumerate(self.param_history):
            if result.acc > best_acc:
                best_acc_idx = n
                best_acc = result.acc

        return self.param_history[best_acc_idx]

    def get_total_num_params(self) -> int:
        return self.total_num_params

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
        total_num_param = len(lr_range) * len(wd_range) * len(dp_range) * len(mn_range)
        self.total_num_params = total_num_param
        cur_param = 0
        # TODO : prepare all the params ahead of time (eg: into a list) and then run the search?
        # One potential benefit here might be that we can divide the list up
        # and send parts of it to different threads, nodes, etc.
        for lr in lr_range:
            for wd in wd_range:
                for dp in dp_range:
                    for mn in mn_range:
                        print('Searching param combination [%d / %d]' % (cur_param, total_num_param))

                        if lr != 0.0:
                            self.trainer.set_learning_rate(lr)
                        if wd != 0.0:
                            self.trainer.set_weight_decay(wd)
                        if dp != 0.0:
                            self.trainer.set_dropout(dp)
                        if mn != 0.0:
                            self.trainer.set_momentum(mn)

                        self.trainer.model.init_weights()
                        if not self.dont_train:
                            self.trainer.restart_history()
                            self.trainer.set_num_epochs(self.max_num_epochs)
                            for epoch in range(self.max_num_epochs):
                                #self.trainer.train()
                                self.trainer.train_epoch()
                                self.trainer.val_epoch()

                                cur_acc = self.trainer.get_cur_acc()
                                acc_since = self.trainer.get_acc_since(self.min_num_epochs)

                                if (epoch >= self.min_num_epochs) and (cur_acc > self.min_req_acc):
                                    search_result = SearchResult()
                                    search_result.loss_history   = self.trainer.get_loss_history()
                                    search_result.acc_histoy     = self.trainer.get_acc_history()
                                    search_result.trainer_params = self.trainer.get_trainer_params()
                                    search_result.acc            = self.trainer.get_best_acc()
                                    # Copy the params that produced the result
                                    search_params = {k: 0 for k in params.keys()}
                                    search_params['learning_rate'] = lr
                                    search_params['weight_decay']  = wd
                                    search_params['dropout']       = dp
                                    search_params['momentum']      = mn
                                    search_result.params = search_params
                                    self.param_history.append(search_result)

                                    if self.verbose:
                                        print('Found grid result :\n %s' % str(search_result))

                                    break
                                elif (epoch >= self.min_num_epochs) and (acc_since < self.min_req_acc):
                                    if self.verbose:
                                        print('Parameter set %d/%d failed to acheive acc > %f in %d epochs' %\
                                              (cur_param, total_num_param, self.min_req_acc, epoch)
                                        )
                                    break

                                self.trainer.cur_epoch += 1

                        cur_param += 1

        # reset the trainer params
        self.trainer.set_trainer_params(self.original_trainer_params)




class RandomSearcher(object):
    """
    RandomSearcher
    Module to perform random searches over hyperparameter space.
    """
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
        self.num_params:int     = kwargs.pop('num_params', 8)
        # cycle parameters
        self.max_num_epochs:int = kwargs.pop('max_num_epochs', 10)
        self.min_num_epochs:int = kwargs.pop('min_num_epochs', 4)
        self.min_req_acc:float  = kwargs.pop('min_req_acc', 0.4)    # reject acc less than this after min_num_epochs

        self.original_trainer_params = dict()

    def __repr__(self) -> str:
        return 'RandomSearcher'

    def _init_history(self) -> None:
        self.param_history = []

    def save_history(self, prefix:str=None) -> None:
        for n, result in enumerate(self.param_history):
            if prefix is not None:
                fname = prefix + '_search_result_' + str(n) + '.pkl'
            else:
                fname = 'search_result' + str(n) + '.pkl'
            with open(fname, 'w') as fp:
                pickle.dumps(result)

    def search(self, params:dict) -> None:
        # I seem to remember something about using logarithmic params
        if 'learning_rate' in params:
            lr_range = 10 * np.random.uniform(
                params['learning_rate'][0],
                params['learning_rate'][1],
                self.num_params
            )
        else:
            lr_range = [0.0]

        if 'weight_decay' in params:
            wd_range = 10 ** np.random.uniform(
                params['weight_decay'][0],
                params['weight_decay'][1],
                self.num_params
            )
        else:
            wd_range = [0.0]

        if 'dropout' in params:
            dp_range = 10 ** np.random.uniform(
                params['dropout'][0],
                params['dropout'][1],
                self.num_params
            )
        else:
            dp_range = [0.0]

        if 'momentum' in params:
            mn_range = 10 ** np.random.uniform(
                params['momentum'][0],
                params['momentum'][1],
                self.num_params
            )
        else:
            mn_range = [0.0]

        self.original_trainer_params = self.trainer.get_trainer_params()
