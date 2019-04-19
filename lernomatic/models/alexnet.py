"""
ALEXNET
Various versions of Alexnet, should that be needed for something

Stefan Wong 2019
"""

import torch
import torch.nn as nn
from lernomatic.models import common


class AlexNetCIFAR10(common.LernomaticModel):
    def __init__(self) -> None:
        self.net = AlexNetModule(10)
        self.model_name = 'AlexNetCIFAR10'
        self.module_name = 'AlexNet'
        self.import_path = 'lernomatic.models.alexnet'
        self.module_import_path = 'lernomatic.models.alexnet'

    def __repr__(self) -> str:
        return 'AlexNetCIFAR10'


class AlexNetModule(nn.Module):
    def __init__(self, num_classes) -> None:
        super(AlexNetModule, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x
