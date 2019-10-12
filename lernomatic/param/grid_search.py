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
        self.param:dict = param
        self.acc:float = acc

    def __repr__(self) -> str:
        return 'GridResult'



class GridSearcher(object):
    def __init__(self,
                 model:common.LernomaticModel=None,
                 tr:trainer.Trainer=None,
                 **kwargs) -> None:
        valid_params = ('learning_rate', 'weight_decay', 'dropout', 'momentum')
        self.model = model
        self.trainer = tr
        self.params:dict = kwargs.pop('params', None)

        # how many params to search over
        self.num_params:int = kwargs.pop('num_params', 8)
        # cycle parameters
        self.num_epochs:int = kwargs.pop('num_epochs', 10)


    def __repr__(self) -> str:
        return 'GridSearcher'

    def _init_history(self) -> None:
        pass

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

        self.trainer.set_num_epochs(self.num_epochs)

        # Do the actual grid search
        total_params = len(lr_range) + len(wd_range) + len(dp_range) + len(mn_range)
        cur_param = 0
        for lr in lr_range:
            for wd in wd_range:
                for dp in dp_range:
                    for mn in mn_range:
                        print('Searching param [%d / %d]' % (cur_param, total_params))

                        if lr != 0.0:
                            self.trainer.set_learning_rate(lr)
                        if wd != 0.0:
                            self.trainer.set_weight_decay(wd)
                        if dp != 0.0:
                            self.trainer.set_dropout(dp)
                        if mn != 0.0:
                            self.trainer.set_momentum(mn)

                        self.trainer.train()

                        cur_param += 1

