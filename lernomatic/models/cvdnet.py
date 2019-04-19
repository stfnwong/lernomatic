"""
CVDNET
Network for Cats-vs-Dogs

"""

import torch
import torch.nn as nn
import torchvision
from lernomatic.models import common

# debug
#from pudb import set_trace; set_trace()


class CVDNet(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = CVDNetModule(**kwargs)
        self.module_name = 'CVDNet'
        self.model_name = 'CVDNet'
        self.import_path = 'lernomatic.models.cvdnet'
        self.module_import_path = 'lernomatic.models.cvdnet'

    def __repr__(self) -> str:
        return 'CVDNet'


class CVDNet2(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = CVDNet2Module(**kwargs)
        self.module_name = 'CVDNet2'
        self.model_name = 'CVDNet2'
        self.import_path = 'lernomatic.models.cvdnet'
        self.module_import_path = 'lernomatic.models.cvdnet'

    def __repr__(self) -> str:
        return 'CVDNet'


class CVDNetModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.num_trim_layers = kwargs.pop('num_trim_layers', 2)
        super(CVDNet, self).__init__()
        sub_model = torchvision.models.resnet34(pretrained=True)

        # TODO: make this settable
        for param in sub_model.parameters():
            param.requires_grad = False
        num_features = sub_model.fc.in_features
        sub_model.fc = nn.Linear(num_features, 2)

        #modules = list(sub_model.children())[:-self.num_trim_layers]
        modules = list(sub_model.children())
        self.net = nn.Sequential(*modules)
        self.sig = nn.Sigmoid()

    def forward(self, X) -> torch.Tensor:
        out = self.net(X)
        #out = self.fc(out)
        out = self.sig(out)

        return out



class CVDNet2Module(nn.Module):
    def __init__(self, **kwargs):
        super(CVDNet2Module, self).__init__()
        self.sub_model = torchvision.models.resnet34(pretrained=True)

        for param in self.sub_model.parameters():
            param.required_grad = False

        nf = self.sub_model.fc.in_features
        self.sub_model.fc = nn.Linear(nf, 2)
        #self.sig = nn.Sigmoid()
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, X) -> torch.Tensor:
        out = self.sub_model(X)
        #out = self.sig(out)
        #out = self.softmax(out)

        return out
