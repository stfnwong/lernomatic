"""
PRETRAINED
LernomaticModel wrappers for Torchvision models

Stefan Wong 2019
"""

import torch
import torch.nn as nn
import torchvision
from lernomatic.models import common


class AlexNet(common.LernomaticModel):
    def __init__(self) -> None:
        self.model_name = 'AlexNet'
        self.module_name = 'AlexNetModule'
        self.import_path = 'lernomatic.models.pretrained'
        self.module_import_path = 'lernomatic.models.pretrained'
        self.net = AlexNetModule()

    def __repr__(self) -> str:
        return 'AlexNet'

    def get_num_layers(self) -> int:
        n = 0
        for param in self.net.parameters():
            n += 1
        return n


class AlexNetModule(nn.Module):
    def __init__(self) -> None:
        super(AlexNetModule, self).__init__()
        net = torchvision.models.alexnet(pretrained=True)
        net_modules = list(net.children())
        self.net = nn.Sequential(*net_modules)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.net(X)
