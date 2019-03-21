"""
COMMON
Contains definitions for base model classes in this library

Stefan Wong 2018
"""

import torch
import importlib


class LernomaticModel(object):
    def __init__(self, **kwargs) -> None:
        self.net               : torch.nn.Module = None
        self.import_path       : str             = 'lernomatic.model.common'
        self.model_name        : str             = 'LernomaticModel'
        self.module_name       : str             = None
        self.module_import_path: str             = None

    def __repr__(self) -> str:
        return 'LernomaticModel'

    def get_model_parameters(self) -> dict:
        return self.net.parameters()

    def get_params(self) -> dict:
        params = {
            'model_state_dict'   : self.net.state_dict(),
            'model_name'         : self.get_model_name(),
            'model_import_path'  : self.get_model_path(),
            'module_name'        : self.get_module_name(),
            'module_import_path' : self.get_module_import_path()
        }
        return params

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_path(self) -> str:
        return self.import_path

    def get_module_name(self) -> str:
        return self.module_name

    def get_module_import_path(self) -> str:
        return self.module_import_path

    def get_net(self) -> torch.nn.Module:
        if self.net is None:
            raise ValueError('No network set in module %s' % repr(self))
        return self.net

    def get_model_state_dict(self) -> dict:
        if self.net is None:
            raise ValueError('No network set in module %s' % repr(self))
        return self.net.state_dict()

    def set_params(self, params : dict) -> None:
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)
        self.net = mod()
        self.net.load_state_dict(params['model_state_dict'])

    def set_train(self) -> None:
        if self.net is None:
            raise ValueError('No network set in module %s' % repr(self))
        self.net.train()

    def set_eval(self) -> None:
        if self.net is None:
            raise ValueError('No network set in module %s' % repr(self))
        self.net.eval()

    def set_net_state_dict(self, sd: dict) -> None:
        self.net.state_dict(sd)

    def send_to(self, device : torch.device) -> None:
        if self.net is None:
            raise ValueError('No network set in module %s' % repr(self))
        self.net.to(device)

    def forward(self, X) -> torch.Tensor:
        if self.net is None:
            raise ValueError('No network set in module %s' % repr(self))
        return self.net(X)
