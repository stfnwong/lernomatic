"""
DCGAN
Modules for DCGAN

"""

import torch
import torch.nn as nn
from lernomatic.models import common

# debug
#from pudb import set_trace; set_trace()


# This is a direct implementation of the Generator and Discriminator
# architectures from "Unsupervised Representation Learning..." (ArXiV :
# 1511.06434v2)

class DCGGenerator(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = DCGGeneratorModule(**kwargs)
        self.model_name         = 'DCGGenerator'
        self.module_name        = 'DCGGeneratorModule'
        self.import_path        = 'lernomatic.models.gan.dcgan_basic'
        self.module_import_path = 'lernomatic.models.gan.dcgan_basic'

    def __repr__(self) -> str:
        return 'DCGGenerator'

    def get_zvec_dim(self) -> int:
        return self.net.zvec_dim

    def get_num_filters(self) -> int:
        return self.net.num_filters

    def zero_grad(self) -> None:
        self.net.zero_grad()

    def init_weights(self) -> None:
        classname = self.net.__class__.__name__
        if classname.find('Conv') != -1:
            self.net.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            self.net.weight.data.normal_(1.0, 0.02)
            self.net.bias.data.fill_(0)

# Generator implementation
class DCGGeneratorModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(DCGGeneratorModule, self).__init__()
        self.zvec_dim     = kwargs.pop('zvec_dim', 100)         # this is the size of zdim in Metz and Chintala (2016)
        self.num_filters  = kwargs.pop('num_filters', 64)
        self.num_channels = kwargs.pop('num_channels', 3)     # number of channels in output image

        # z is input going into covolution
        self.convtranspose1 = nn.ConvTranspose2d(
            self.zvec_dim,
            self.num_filters * 8,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False)
        self.bn1   = nn.BatchNorm2d(self.num_filters * 8)
        self.relu1 = nn.ReLU(True)
        # state size, (num_filters * 8) x 4 x 4
        self.convtranspose2 = nn.ConvTranspose2d(
            self.num_filters * 8,
            self.num_filters * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.bn2   = nn.BatchNorm2d(self.num_filters * 4)
        self.relu2 = nn.ReLU(True)
        # state size (num_filters * 4) x 8 x 8
        self.convtranspose3 = nn.ConvTranspose2d(
            self.num_filters * 4,
            self.num_filters * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(self.num_filters * 2)
        self.relu3 = nn.ReLU(True)
        # state size (num_filters*2) x 16 x 16
        self.convtranspose4 = nn.ConvTranspose2d(
            self.num_filters * 2,
            self.num_filters,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.bn4 = nn.BatchNorm2d(self.num_filters)
        self.relu4 = nn.ReLU(True)
        # state size (num_filters) x 32 x 32
        self.convtranspose5 = nn.ConvTranspose2d(
            self.num_filters,
            self.num_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.tanh = nn.Tanh()
        # state size (num_channels) x 64 x 64

    def forward(self, X) -> torch.Tensor:
        # block 1
        out = self.convtranspose1(X)
        out = self.bn1(out)
        out = self.relu1(out)

        # block 2
        out = self.convtranspose2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # block 3
        out = self.convtranspose3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        # block 4
        out = self.convtranspose4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = self.convtranspose5(out)
        out = self.tanh(out)

        return out


class DCGDiscriminator(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = DCGDiscriminatorModule(**kwargs)
        # internal bookkeeping
        self.model_name         = 'DCGDiscriminator'
        self.module_name        = 'DCGDiscriminatorModule'
        self.import_path        = 'lernomatic.models.gan.dcgan_basic'
        self.module_import_path = 'lernomatic.models.gan.dcgan_basic'

    def __repr__(self) -> str:
        return 'DCGDiscriminator'

    def get_num_filters(self) -> int:
        return self.net.num_filters

    def zero_grad(self) -> None:
        self.net.zero_grad()

    def init_weights(self) -> None:
        pass



# TODO : split the sequential into distinct layers (so that we can step
# through with debugger)
class DCGDiscriminatorModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(DCGDiscriminatorModule, self).__init__()
        self.num_filters  = kwargs.pop('num_filters', 64)
        self.num_channels = kwargs.pop('num_channels', 3)

        # input is (num_channels) x 64 x 64
        self.conv1 = nn.Conv2d(
            self.num_channels,
            self.num_filters,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        # state size (num_filters) x 32 x 32
        self.conv2 = nn.Conv2d(
            self.num_filters,
            self.num_filters * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(self.num_filters * 2)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        # state size (num_filters * 2) x 16 x 16
        self.conv3 = nn.Conv2d(
            self.num_filters * 2,
            self.num_filters * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(self.num_filters * 4)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        # state size (num_filters * 4) x 8 x 8
        self.conv4 = nn.Conv2d(
            self.num_filters * 4,
            self.num_filters * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.bn4 = nn.BatchNorm2d(self.num_filters * 8)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        # state size (num_filters * 8) x 4 x 4
        self.conv5 = nn.Conv2d(
            self.num_filters * 8,
            1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X) -> torch.Tensor:
        # block 1
        out = self.conv1(X)
        out = self.lrelu1(out)

        # block 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu2(out)

        # block 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.lrelu3(out)

        # block 4
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.lrelu4(out)

        out = self.conv5(out)
        out = self.sigmoid(out)

        return out
