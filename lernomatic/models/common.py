"""
COMMON
Contains definitions for base model classes in this library

Stefan Wong 2018
"""

import torch
import importlib


class LernomaticModel(object):
    """
    LERNOMATICMODEL
    Base class for models in Lernomatic.

    This object wraps a torch.nn.Module. Its main purpose is to provide a consistent
    interface for loading checkpoints, model weights, and so on. This is done by
    documenting the paths and names of class attributes and later instantiating them
    with importlib.
    """
    def __init__(self, **kwargs) -> None:
        self.net               : torch.nn.Module = None
        self.import_path       : str             = 'lernomatic.model.common'
        self.model_name        : str             = 'LernomaticModel'
        self.module_name       : str             = None
        self.module_import_path: str             = None

    def __repr__(self) -> str:
        return 'LernomaticModel'

    def get_model_parameters(self) -> dict:
        """
        Returns torch model parameters (state_dict)
        """
        return self.net.parameters()

    def get_num_layers(self) -> int:
        n = 0
        for l, param in enumerate(self.net.parameters()):
            n += 1
        return n

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

    def get_net_state_dict(self) -> dict:
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
        """
        Send the inner module(s) to a given device
        """
        if self.net is None:
            raise ValueError('No network set in module %s' % repr(self))
        self.net.to(device)

    def forward(self, X) -> torch.Tensor:
        """
        Wraps the forward pass of the inner module(s)
        """
        if self.net is None:
            raise ValueError('No network set in module %s' % repr(self))
        return self.net(X)

    # Layer freezing
    def freeze_to(self, N:int) -> None:
        """
        FREEZE_TO
        Freeze the first N layers of the model
        """
        for l, param in enumerate(self.net.parameters()):
            param.requires_grad = False
            if l >= N:
                break

    def unfreeze_to(self, N:int) -> None:
        """
        UNFREEZE_TO
        Freeze the first N layers of the model
        """
        for l, param in enumerate(self.net.parameters()):
            param.requires_grad = True
            if l >= N:
                break

    def freeze(self) -> None:
        """
        FREEZE
        Freeze all but the last layer of the module
        """
        num_layers = self.get_num_layers()
        for l, param in enumerate(self.net.parameters()):
            param.requires_grad = False
            if l == num_layers-1:
                break

    def freeze_all(self) -> None:
        """
        FREEZE
        Freeze all layers including the last layer of the module
        """
        for l, param in enumerate(self.net.parameters()):
            param.requires_grad = False

    def unfreeze(self) -> None:
        for l, param in enumerate(self.net.parameters()):
            param.requires_grad = True

    # Load the model component directly from a checkpoint
    def load_checkpoint(self, fname, model_key='model'):
        """
        load_checkpoint
        Load model information from a trainer checkpoint file
        """
        checkpoint_data = torch.load(fname)
        model_params = dict()
        model_params.update({'model_state_dict' : checkpoint_data[model_key]['model_state_dict']})
        model_params.update({'model_name' : checkpoint_data[model_key]['mode_name']})
        model_params.update({'model_import_path' : checkpoint_data[model_key]['model_import_path']})
        model_params.update({'module_name' : checkpoint_data[model_key]['module_name']})
        model_params.update({'module_import_path' : checkpoint_data[model_key]['module_import_path']})
        self.set_params(model_params)

    def zero_grad(self) -> None:
        self.net.zero_grad()
