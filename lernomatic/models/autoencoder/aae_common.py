"""
AAE_COMMON
Some basic AAEencoder models

Stefan Wong 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lernomatic.models import common


# debug
#from pudb import set_trace; set_trace()


# Encoder side modules
class AAEQNet(common.LernomaticModel):
    def __init__(self,
                 x_dim:int,
                 z_dim:int,
                 hidden_size:int,
                 dropout:float=0.2) -> None:
        self.import_path       : str             = 'lernomatic.models.autoencoder.aae_common'
        self.model_name        : str             = 'AAEQNet'
        self.module_name       : str             = 'AAEQNetModule'
        self.module_import_path: str             = 'lernomatic.models.autoencoder.aae_common'
        self.net = AAEQNetModule(
            x_dim,
            z_dim,
            hidden_size,
            dropout=dropout
        )

    def __repr__(self) -> str:
        return 'AAEQNet'

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_x_dim(self) -> int:
        return self.net.x_dim

    def get_z_dim(self) -> int:
        return self.net.z_dim


class AAEQNetModule(nn.Module):
    def __init__(self,
                 x_dim:int,
                 z_dim:int,
                 hidden_size:int,
                 dropout:float=0.2) -> None:
        self.x_dim       :int = x_dim
        self.z_dim       :int = z_dim
        self.hidden_size :int = hidden_size
        self.dropout     :float = dropout

        super(AAEQNetModule, self).__init__()

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
class AAEPNet(common.LernomaticModel):
    def __init__(self, x_dim:int, z_dim:int, hidden_size:int, dropout:float=0.2) -> None:
        self.import_path       : str             = 'lernomatic.models.autoencoder.aae_common'
        self.model_name        : str             = 'AAEPNet'
        self.module_name       : str             = 'AAEPNetModule'
        self.module_import_path: str             = 'lernomatic.models.autoencoder.aae_common'
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


class AAEPNetModule(nn.Module):
    def __init__(self, x_dim:int, z_dim:int, hidden_size:int, dropout:float=0.2) -> None:
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.hidden_size = hidden_size
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
    def __init__(self, z_dim:int, hidden_size:int, dropout:float=0.2) -> None:
        self.import_path       : str             = 'lernomatic.models.autoencoder.aae_common'
        self.model_name        : str             = 'AAEDNetGauss'
        self.module_name       : str             = 'AAEDNetGaussModule'
        self.module_import_path: str             = 'lernomatic.models.autoencoder.aae_common'
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

