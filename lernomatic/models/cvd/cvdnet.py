"""
CVDNET
Network for Cats-vs-Dogs

"""

import torch
import torch.nn as nn
import torchvision
from lernomatic.models import common

# debug
#

class CVDNet(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = CVDNetModule(**kwargs)
        self.import_path = 'lernomatic.models.cvd.cvdnet'
        self.model_name = 'CVDNet'
        self.module_name = 'CVDNetModule'
        self.module_import_path = 'lernomatic.models.cvd.cvdnet'

    def __repr__(self) -> str:
        return 'CVDNet'


class CVDNetModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.num_trim_layers = kwargs.pop('num_trim_layers', 2)
        super(CVDNetModule, self).__init__()
        sub_model = torchvision.models.resnet34(pretrained=True)

        # TODO: make this settable
        for param in sub_model.parameters():
            param.requires_grad = False
        num_features = sub_model.fc.in_features
        sub_model.fc = nn.Linear(num_features, 2)

        modules = list(sub_model.children())
        self.net = nn.Sequential(*modules)
        self.sig = nn.Sigmoid()

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.net(X)
        out = self.sig(out)
        return out



class CVDNet2(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = CVDNet2Module(**kwargs)
        self.import_path = 'lernomatic.models.cvd.cvdnet'
        self.model_name = 'CVDNet2'
        self.module_name = 'CVDNet2Module'
        self.module_import_path = 'lernomatic.models.cvd.cvdnet'

    def __repr__(self) -> str:
        return 'CVNet2'


class CVDNet2Module(nn.Module):
    def __init__(self, **kwargs):
        super(CVDNet2Module, self).__init__()
        self.sub_model = torchvision.models.resnet34(pretrained=True)

        for param in self.sub_model.parameters():
            param.required_grad = False

        nf = self.sub_model.fc.in_features
        self.sub_model.fc = nn.Linear(nf, 2)

    def forward(self, X):
        return self.sub_model(X)
