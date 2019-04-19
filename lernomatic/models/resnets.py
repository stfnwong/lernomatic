"""
RESNETS
Modules for building resnets

Stefan Wong 2019
"""

import importlib
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lernomatic.models import common

# debug
#from pudb import set_trace; set_trace()


class ResnetBlock(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 stride: int,
                 drop_rate:float=0.0) -> None:
        super(ResnetBlock, self).__init__()
        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size = 3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size = 3,
            stride=1,
            padding=1,
            bias=False
        )

        if self.equal_in_out:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size = 1,
                    stride = stride,
                    padding = 0,
                    bias = False
                ),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, X) -> torch.Tensor:
        out = self.bn1(X)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        # resnet shortcut
        out += self.shortcut(X)

        return out


class NetworkBlock(nn.Module):
    def __init__(self,
                 num_layers: int,
                 in_planes: int,
                 out_planes: int,
                 block : nn.Module,
                 stride: int,
                 drop_rate=0.0) -> None:
        super(NetworkBlock, self).__init__()
        # Create layers
        layers = []
        for i in range(num_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    drop_rate,
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, X) -> torch.Tensor:
        return self.layers(X)


class WideResnetModule(nn.Module):
    def __init__(self,
                 depth:int,
                 num_classes:int,
                 input_channels:int=3,
                 w_factor:int=1,
                 drop_rate=0.0) -> None:
        super(WideResnetModule, self).__init__()

        self.depth = depth      # for __str___()
        num_channels = [16, 16 * w_factor, 32 * w_factor, 64 * w_factor]
        if (depth-4) % 6 != 0:
            raise ValueError('depth-4 must be divisible by 6 (current depth = %d' % depth)
        n = int((depth - 4) / 6)

        # first conv layer before  any network block
        self.conv1 = nn.Conv2d(
            input_channels,
            num_channels[0],
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )
        # first resnet block
        self.block1 = NetworkBlock(
            n,
            num_channels[0],
            num_channels[1],
            ResnetBlock,
            1,
            drop_rate
        )
        # second resnet block
        self.block2 = NetworkBlock(
            n,
            num_channels[1],
            num_channels[2],
            ResnetBlock,
            2,
            drop_rate
        )
        # third resnet block
        self.block3 = NetworkBlock(
            n,
            num_channels[2],
            num_channels[3],
            ResnetBlock,
            2,
            drop_rate
        )
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_channels[3], num_classes)
        self.num_channels = num_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def __str__(self) -> str:
        s = []
        s.append('WideResnet-%d\n' % self.depth)
        s.append(str(self))

        return ''.join(s)

    def forward(self, X) -> torch.Tensor:
        out = self.conv1(X)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)

        return self.fc(out)


class WideResnet(common.LernomaticModel):
    def __init__(self, **kwargs):
        self.depth          :int   = kwargs.pop('depth', 58)
        self.num_classes    :int   = kwargs.pop('num_classes', 10)
        self.input_channels :int   = kwargs.pop('input_channels', 3)
        self.w_factor       :int   = kwargs.pop('w_factor', 1)
        self.drop_rate      :float = kwargs.pop('drop_rate', 0.0)

        self.net = WideResnetModule(
            self.depth,
            self.num_classes,
            self.input_channels,
            self.w_factor,
            self.drop_rate
        )
        self.model_name = 'WideResnet'
        self.module_name = 'WideResnetModule'
        self.import_path = 'lernomatic.models.resnets'
        self.module_import_path = 'lernomatic.models.resnets'

    def __repr__(self) -> str:
        return 'WideResnet'

    def get_params(self) -> dict:
        params = super(WideResnet, self).get_params()
        params['resnet_params'] = {
            'depth'          : self.depth,
            'num_classes'    : self.num_classes,
            'input_channels' : self.input_channels,
            'w_factor'       : self.w_factor,
            'drop_rate'      : self.drop_rate
        }

        return params

    def set_params(self, params : dict) -> None:
        # load resnet params
        self.depth          = params['resnet_params']['depth']
        self.num_classes    = params['resnet_params']['num_classes']
        self.input_channels = params['resnet_params']['input_channels']
        self.w_factor       = params['resnet_params']['w_factor']
        self.drop_rate      = params['resnet_params']['drop_rate']
        # regular model stuff
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)
        self.net = mod(
            self.depth,
            self.num_classes,
            self.input_channels,
            self.w_factor,
            self.drop_rate
        )
        self.net.load_state_dict(params['model_state_dict'])

