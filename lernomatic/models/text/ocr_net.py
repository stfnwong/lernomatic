"""
OCR_NET
Network for Optical Character Recognition (eg: im2latex)

Stefan Wong 2019
"""

import torch
import torch.nn as nn
from lernomatic.models import common


class OCRNet(common.LernomaticModel):
    """
    Network for Optical Character Recognition.
    The structure for the module is taken from im2latex examples (TODO: add the actual link)
    """
    def __init__(self, **kwargs) -> None:
        self.net = OCRNetModule(**kwargs)
        self.import_path       : str             = 'lernomatic.models.text'
        self.model_name        : str             = 'OCRNet'
        self.module_name       : str             = 'OCRNetModule'
        self.module_import_path: str             = 'lernomatic.models.text'

    def __repr__(self) -> str:
        return 'OCRNet'



class OCRNetModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.kernel_size:int        = kwargs.pop('kernel_size', 3)
        self.num_input_channels:int = kwargs.pop('num_input_channels', 3)
        self.enc_out_dim:int        = kwargs.pop('enc_out_dim', 512)

        super(OCRNetModule, self).__init__()

        # Create network graph
        self.conv1 = nn.Conv2d(self.num_input_channels, 64, kernel_size = self.kernel_size, stride=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad=1)

        self.conv2 = nn.Conv2d(64, 128, kernel_size = self.kernel_size, stride=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad=1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size = self.kernel_size, stride=1, bias=False)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size = self.kernel_size, stride=1, bias=False)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(512, 512, kernel_size = self.kernel_size, stride=1, bias=False)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.conv6 = nn.Conv2d(512, self.enc_out_dim, kernel_size = self.kernel_size, stride=1, pad=0)
        self.relu6 = nn.ReLU(inplace=True)


    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.conv1(X)
        out = self.relu1(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        # mid block
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.relu4(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.pool5(out)

        out = self.conv6(out)
        out = self.relu6(out)

        return out

