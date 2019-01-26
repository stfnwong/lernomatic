"""
RESNETS
Modules for building resnets

Stefan Wong 2019
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# debug
from pudb import set_trace; set_trace()

class ResnetBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
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

    def forward(self, X):
        out = self.bn1(X)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)

        #if self.downsample is not None:
        #    residual = self.downsample(X)   # TODO : issue here with tensor size...
        #out += residual
        out += self.shortcut(X)

        return out


class NetworkBlock(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super(NetworkBlock, self).__init__()
        # Create layers
        layers = []
        #downsample = None
        #if (stride != 1) or (in_planes != out_planes):
        #    downsample = nn.Sequential(
        #        nn.Conv2d(
        #            in_planes,
        #            out_planes,
        #            kernel_size=3,
        #            stride=stride,
        #            padding=1,
        #            bias=False
        #        ),
        #        nn.BatchNorm2d(out_planes)
        #    )
        for i in range(num_layers):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    drop_rate,
                    #downsample
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, X):
        return self.layers(X)


class WideResnet(nn.Module):
    def __init__(self, depth, num_classes, w_factor=1, drop_rate=0.0):
        super(WideResnet, self).__init__()

        num_channels = [16, 16 * w_factor, 32 * w_factor, 64 * w_factor]
        assert((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)

        # first conv layer before  any network block
        self.conv1 = nn.Conv2d(
            3,
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

    def forward(self, X):
        out = self.conv1(X)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)

        return self.fc(out)
