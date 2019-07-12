"""
AAE_COMMON
Some basic AAEencoder models

Stefan Wong 2019
"""

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from lernomatic.models import common

# debug
#from pudb import set_trace; set_trace()


# Encoder side modules
class AAEQNet(common.LernomaticModel):
    def __init__(self,
                 x_dim:int=784,
                 z_dim:int=2,
                 hidden_size:int=512,
                 num_classes:int=10,
                 dropout:float=0.2,
                 cat_mode:bool=False) -> None:
        self.import_path       : str = 'lernomatic.models.autoencoder.aae_common'
        self.model_name        : str = 'AAEQNet'
        self.module_name       : str = 'AAEQNetModule'
        self.module_import_path: str = 'lernomatic.models.autoencoder.aae_common'
        self.net = AAEQNetModule(
            x_dim,
            z_dim,
            hidden_size,
            num_classes = num_classes,
            dropout=dropout,
            cat_mode = cat_mode
        )

    def __repr__(self) -> str:
        return 'AAEQNet'

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_x_dim(self) -> int:
        return self.net.x_dim

    def get_z_dim(self) -> int:
        return self.net.z_dim

    def get_num_classes(self) -> int:
        return self.net.num_classes

    def set_cat_mode(self) -> None:
        self.net.cat_mode = True

    def unset_cat_mode(self) -> None:
        self.net.cat_mode = False

    def get_model_args(self) -> dict:
        return {
            'x_dim'       : self.net.x_dim,
            'z_dim'       : self.net.z_dim,
            'hidden_size' : self.net.hidden_size,
            'num_classes' : self.net.num_classes,
            'dropout'     : self.net.dropout,
            'cat_mode'    : self.net.cat_mode
        }

    def set_params(self, params : dict) -> None:
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)

        self.net = mod(
            params['model_args']['x_dim'],
            params['model_args']['z_dim'],
            params['model_args']['hidden_size'],
            dropout = params['model_args']['dropout'],
            cat_mode = params['model_args']['cat_mode']
        )
        self.net.load_state_dict(params['model_state_dict'])


class AAEQNetModule(nn.Module):
    def __init__(self,
                 x_dim:int,
                 z_dim:int,
                 hidden_size:int,
                 num_classes:int=10,
                 dropout:float=0.2,
                 cat_mode:bool=False) -> None:
        self.x_dim       :int = x_dim
        self.z_dim       :int = z_dim
        self.hidden_size :int = hidden_size
        self.num_classes :int = num_classes
        self.dropout     :float = dropout
        self.cat_mode    :bool  = cat_mode

        super(AAEQNetModule, self).__init__()

        # network graph
        self.l1 = nn.Linear(self.x_dim, self.hidden_size)           # MNIST size?
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        # gaussian (z)
        self.lingauss = nn.Linear(self.hidden_size, self.z_dim)
        # categorical code (y)
        self.lincat = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = self.l1(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        X = self.l2(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        xgauss = self.lingauss(X)

        if self.cat_mode:
            xcat = F.softmax(self.lincat(X), dim=0)
            return (xcat, xgauss)

        return xgauss



# Decoder side
class AAEPNet(common.LernomaticModel):
    def __init__(self,
                 x_dim:int=784,
                 z_dim:int=2,
                 hidden_size:int=512,
                 dropout:float=0.2) -> None:
        self.import_path       : str = 'lernomatic.models.autoencoder.aae_common'
        self.model_name        : str = 'AAEPNet'
        self.module_name       : str = 'AAEPNetModule'
        self.module_import_path: str = 'lernomatic.models.autoencoder.aae_common'
        self.net = AAEPNetModule(
            x_dim,
            z_dim,
            hidden_size,
            dropout=dropout
        )

    def __repr__(self) -> str:
        return 'AAEPNet'

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_x_dim(self) -> int:
        return self.net.x_dim

    def get_z_dim(self) -> int:
        return self.net.z_dim

    def get_model_args(self) -> dict:
        return {
            'x_dim'       : self.net.x_dim,
            'z_dim'       : self.net.z_dim,
            'hidden_size' : self.net.hidden_size,
            'dropout'     : self.net.dropout
        }

    def set_params(self, params : dict) -> None:
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)

        self.net = mod(
            params['model_args']['x_dim'],
            params['model_args']['z_dim'],
            params['model_args']['hidden_size'],
            dropout = params['model_args']['dropout'],
        )
        self.net.load_state_dict(params['model_state_dict'])


class AAEPNetModule(nn.Module):
    def __init__(self, x_dim:int, z_dim:int, hidden_size:int, dropout:float=0.2) -> None:
        self.x_dim       :int   = x_dim
        self.z_dim       :int   = z_dim
        self.hidden_size :int   = hidden_size
        self.dropout     :float = dropout

        super(AAEPNetModule, self).__init__()
        # network graph
        self.l1 = nn.Linear(self.z_dim, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, self.x_dim)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = self.l1(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        X = self.l2(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        X = self.l3(X)
        X = F.sigmoid(X)

        return X


class AAEDNetGauss(common.LernomaticModel):
    def __init__(self,
                 z_dim:int=2,
                 hidden_size:int=512,
                 dropout:float=0.2) -> None:
        self.import_path       : str = 'lernomatic.models.autoencoder.aae_common'
        self.model_name        : str = 'AAEDNetGauss'
        self.module_name       : str = 'AAEDNetGaussModule'
        self.module_import_path: str = 'lernomatic.models.autoencoder.aae_common'
        self.net = AAEDNetGaussModule(
            z_dim,
            hidden_size,
            dropout=dropout
        )

    def __repr__(self) -> str:
        return 'AAEDNetGauss'

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_z_dim(self) -> int:
        return self.net.z_dim

    def get_model_args(self) -> dict:
        return {
            'z_dim'       : self.net.z_dim,
            'hidden_size' : self.net.hidden_size,
            'dropout'     : self.net.dropout
        }

    def set_params(self, params : dict) -> None:
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)

        self.net = mod(
            params['model_args']['z_dim'],
            params['model_args']['hidden_size'],
            params['model_args']['dropout'],
        )
        self.net.load_state_dict(params['model_state_dict'])


class AAEDNetGaussModule(nn.Module):
    def __init__(self, z_dim:int, hidden_size:int, dropout:float=0.2) -> None:
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.dropout     :float = dropout

        super(AAEDNetGaussModule, self).__init__()

        # network graph
        self.l1 = nn.Linear(self.z_dim, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, 1)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = self.l1(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        X = self.l2(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        X = self.l3(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        X = F.sigmoid(X)

        return X
