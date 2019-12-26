"""
SEGNET
Implementation of Segnet (https://arxiv.org/pdf/1511.00561.pdf])

Stefan Wong 2019
"""

import torch
import torch.nn as nn
import torchvision.models as models
from lernomatic.models import common


# TODO : decide if I want to end-to-end train a single model
# or have seperate Encoder and Decoder LernomaticModel networks
class SegNet(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = DCGANGeneratorModule(**kwargs)
        self.model_name         = 'DCGANGenerator'
        self.module_name        = 'DCGANGeneratorModule'
        self.import_path        = 'lernomatic.models.gan.dcgan'
        self.module_import_path = 'lernomatic.models.gan.dcgan'

        self.init_weights()

    def __repr__(self) -> str:
        return 'SegNet'





class SegNetEncoderModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        vgg = models.vgg16(pretrained=True, progress=True)
        # We keep just the first 13 layers
        self.net = nn.Sequential(*list(vgg.classifier.children())[:13])

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.net(X)



class SegNetDecoderModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        pass
