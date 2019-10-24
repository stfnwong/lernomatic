"""
RANDOM_SEARCH
Random hyperparameter search

Stefan Wong 2019
"""

import numpy as np
from lernomatic.train import trainer
from lernomatic.models import common



class RandomSearcher(object):
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
