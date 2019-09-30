"""
GAN_UTIL
Various utils for GAN models

Stefan Wong 2019
"""

import torch
import torch.nn as nn


def weight_init(model:nn.Module) -> None:
    # The orignal GAN paper suggests that the weights should be initialized
    # randomly from a normal distribution with mean=0, std=0.02
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def clamp(Xin:torch.Tensor, rot:float=0.1, max_scale:float=1.2) -> torch.Tensor:

    x_s = Xin.select(1, 0)
    x_r = Xin.select(1, 1)
    x_t = Xin.select(1, 2)

    y_s = Xin.select(1, 3)
    y_r = Xin.select(1, 4)
    y_t = Xin.select(1, 5)

    # clamp
    x_s_clamp = torch.unsqueeze(x_s.clamp(max_scale, 2 * max_scale), 1)
    x_r_clamp = torch.unsqueeze(x_r.clamp(-rot, rot), 1)
    x_t_clamp = torch.unsqueeze(x_t.clamp(-1.0, 1.0), 1)

    y_s_clamp = torch.unsqueeze(x_s.clamp(max_scale, 2 * max_scale), 1)
    y_r_clamp = torch.unsqueeze(x_r.clamp(-rot, rot), 1)
    y_t_clamp = torch.unsqueeze(x_t.clamp(-1.0, 1.0), 1)

    t_out = torch.cat([x_s_clamp, x_r_clamp, x_t_clamp, y_r_clamp, y_s_clamp, y_t_clamp], 1)

    return t_out
