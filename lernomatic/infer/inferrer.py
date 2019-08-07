"""
INFERRER
Wraps forward-pass operation for models

Stefan Wong 2019
"""

import torch
import importlib
from lernomatic.models import common

# debug
#from pudb import set_trace; set_trace()

class Inferrer(object):
    """
    Inferrer

    Base class of an Inferrer object. Wraps the forward pass of a model.
    This is the counterpart to the Trainer module. This does the forward
    pass, handles devices, and so on.
    """
    def __init__(self, model=None, **kwargs) -> None:
        self.model = model
        self.device_id:int = kwargs.pop('device_id', -1)

        self._init_device()
        self._send_to_device()

    def __repr__(self) -> str:
        return 'Inferrer'

    def _init_device(self) -> None:
        if self.device_id < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % self.device_id)

    def _send_to_device(self) -> None:
        if self.model is not None:
            self.model.send_to(self.device)

    def get_model(self) -> common.LernomaticModel:
        return self.model

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        return self.model.forward(X)

    def load_model(self, fname:str, model_key='model')-> None:
        self.model = common.LernomaticModel()
        checkpoint_data = torch.load(fname)
        model_params = dict()
        model_params.update({'model_name'         : checkpoint_data[model_key]['model_name']})
        model_params.update({'module_name'        : checkpoint_data[model_key]['module_name']})
        model_params.update({'model_import_path'  : checkpoint_data[model_key]['model_import_path']})
        model_params.update({'module_import_path' : checkpoint_data[model_key]['module_import_path']})
        model_params.update({'model_state_dict'   : checkpoint_data[model_key]['model_state_dict']})

        imp = importlib.import_module(checkpoint_data[model_key]['model_import_path'])
        mod = getattr(imp, checkpoint_data[model_key]['model_name'])
        self.model = mod()
        self.model.set_params(model_params)
        self._send_to_device()
