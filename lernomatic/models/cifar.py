"""
CIFAR10
Model for CIFAR10 example

Stefan Wong 2019
"""

import torchvision
import torch.nn as nn
import torch.nn.functional as F

# A really simple network for testing with CIFAR-10 dataset
class CIFAR10Net(nn.Module):
  def __init__(self):
    super(CIFAR10Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x


# Try CIFAR100 with a pretrained resnet34
class CIFAR100NetR34(nn.Module):
    def __init__(self, grad=False):
        self.net = torchvision.models.resnet56(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = grad

        # update FC section
        nf = self.net.fc.in_features
        self.net.fc = nn.Linear(nf, 100)

    def forward(self, X):
        return self.net(X)
