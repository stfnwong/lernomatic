"""
PROGAN
The GAN from the paper (https://arxiv.org/abs/1710.10196)

Stefan Wong 2020
"""


import torch
import torch.nn as nn
from lernomatic.models import common

from typing import List


def deepcopy_model(module:nn.Module, target:str) -> nn.Sequential:
    new_module = nn.Sequential()
    for name, mod in module.named_children():
        if name == target:
            new_module.add_module(name, m)
            new_module[-1].load_state_dict(m.state_dict())

    return new_module


def get_module_names(model:nn.Module) -> List[str]:
    names = []
    for k, v in model.state_dict().items():
        name = k.split('.')[0]
        if not name in names:
            names.append(name)

    return names


class PixelWiseNormLayer(nn.Module):
    def __init__(self, eps:float=1e-8) -> None:
        super(PixelWiseNormLayer, self).__init__()
        self.eps = eps

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return X / (torch.mean(X**2, dim=1, keepdim=True) + self.eps) ** 0.5


# ================ BLOCKS ================ #
class ProGANUpscale2d(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.scale_factor:int = kwargs.pop('scale_factor', 2)
        super(ProGANUpscale2d, self).__init__()

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        pass



class ProgGANDownscale2d(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(ProGANDownscale2d, self).__init__()

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        xs = X.shape
        X = torch.reshape(X, (-1, xs[1], xs[2], 1, xs[3], 1))

        return X


class ProGANGeneratorBlock(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.halving:bool = kwargs.pop('halving', False)
        self.ndim   :int  = kwargs.pop('ndim', 512)
        self.zvec_dim :int = kwargs.pop('zvec_dim', 512)
        self.norm_latent: bool = kwargs.pop('norm_latent', False)       # TODO; how much extra GPU does this require?



class ProGANGenerator(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = ProGANGeneratorModule(**kwargs)
        self.model_name         = 'ProGANGenerator'
        self.module_name        = 'ProGANGeneratorModule'
        self.import_path        = 'lernomatic.models.gan.progan'
        self.module_import_path = 'lernomatic.models.gan.progan'



class ProGANGeneratorModule(nn.Module):
    pass
