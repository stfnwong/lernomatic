"""
AUTO_COMMON
Some basic Autoencoder models

Stefan Wong 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lernomatic.models import common


# debug
from pudb import set_trace; set_trace()



# Encoder side modules
class AutoQNet(common.LernomaticModel):
    def __init__(self,
                 x_dim:int,
                 z_dim:int,
                 hidden_size:int,
                 dropout:float=0.2) -> None:
        self.import_path       : str             = 'lernomatic.model.autoencoder.auto_common'
        self.model_name        : str             = 'AutoQNet'
        self.module_name       : str             = 'AutoQNetModule'
        self.module_import_path: str             = 'lernomatic.model.autoencoder.auto_common'
        self.net = AutoQNetModule(
            x_dim,
            z_dim,
            hidden_size,
            dropout=dropout
        )

    def __repr__(self) -> str:
        return 'AutoQNet'

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_x_dim(self) -> int:
        return self.net.x_dim

    def get_z_dim(self) -> int:
        return self.net.z_dim


class AutoQNetModule(nn.Module):
    def __init__(self,
                 x_dim:int,
                 z_dim:int,
                 hidden_size:int,
                 dropout:float=0.2) -> None:
        self.x_dim       :int = x_dim
        self.z_dim       :int = z_dim
        self.hidden_size :int = hidden_size
        self.dropout     :float = dropout

        super(AutoQNetModule, self).__init__()

        # network graph
        self.l1 = nn.Linear(self.x_dim, self.hidden_size)           # MNIST size?
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        # gaussian (z)
        self.lingauss = nn.Linear(self.hidden_size, self.z_dim)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = self.l1(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        X = self.l2(X)
        X = F.dropout(X, p=self.dropout)
        X = F.relu(X)
        xgauss = self.lingauss(X)

        return xgauss

# Decoder side
class AutoPNet(common.LernomaticModel):
    def __init__(self, x_dim:int, z_dim:int, hidden_size:int, dropout:float=0.2) -> None:
        self.import_path       : str             = 'lernomatic.model.autoencoder.auto_common'
        self.model_name        : str             = 'AutoPNet'
        self.module_name       : str             = 'AutoPNetModule'
        self.module_import_path: str             = 'lernomatic.model.autoencoder.auto_common'
        self.net = AutoPNetModule(
            x_dim,
            z_dim,
            hidden_size,
            dropout=dropout
        )

    def __repr__(self) -> str:
        return 'AutoPNet'

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_x_dim(self) -> int:
        return self.net.x_dim

    def get_z_dim(self) -> int:
        return self.net.z_dim


class AutoPNetModule(nn.Module):
    def __init__(self, x_dim:int, z_dim:int, hidden_size:int, dropout:float=0.2) -> None:
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.dropout     :float = dropout

        super(AutoPNetModule, self).__init__()
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


class AutoDNetGauss(common.LernomaticModel):
    def __init__(self, z_dim:int, hidden_size:int, dropout:float=0.2) -> None:
        self.import_path       : str             = 'lernomatic.model.autoencoder.auto_common'
        self.model_name        : str             = 'AutoDNetGauss'
        self.module_name       : str             = 'AutoDNetGaussModule'
        self.module_import_path: str             = 'lernomatic.model.autoencoder.auto_common'
        self.net = AutoDNetGaussModule(
            z_dim,
            hidden_size,
            dropout=dropout
        )

    def __repr__(self) -> str:
        return 'AutoDNetGauss'

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_z_dim(self) -> int:
        return self.net.z_dim


class AutoDNetGaussModule(nn.Module):
    def __init__(self, z_dim:int, hidden_size:int, dropout:float=0.2) -> None:
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.dropout     :float = dropout

        super(AutoDNetGaussModule, self).__init__()

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

