"""
MNIST
Model for MNIST example

Stefan Wong 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lernomatic.models import common



class MNISTNet(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = MNISTModule()        # Put **kwargs here if required
        self.import_path = 'lernomatic.models.mnist'
        self.model_name = 'MNISTNet'
        self.module_name = 'MNISTModule'
        self.module_import_path = 'lernomatic.models.mnist'

    def __repr__(self) -> str:
        return 'MNISTNet'


class MNISTModule(nn.Module):
    def __init__(self) -> None:
        super(MNISTModule, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)
