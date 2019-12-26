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

        #self.init_weights()

    def __repr__(self) -> str:
        return 'SegNet'


class SegNetEncoder(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = SegNetEncoderModule(**kwargs)
        self.model_name         = 'SegNetEncoder'
        self.module_name        = 'SegNetEncoderModule'
        self.import_path        = 'lernomatic.models.segmentation.segnet'
        self.module_import_path = 'lernomatic.models.segmentation.segnet'

    def __repr__(self) -> str:
        return 'SegNetEncoder'


class SegNetDecoder(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = SegNetDecoderModule(**kwargs)
        self.model_name         = 'SegNetDecoder'
        self.module_name        = 'SegNetDecoderModule'
        self.import_path        = 'lernomatic.models.segmentation.segnet'
        self.module_import_path = 'lernomatic.models.segmentation.segnet'

    def __repr__(self) -> str:
        return 'SegNetDecoder'



class SegNetEncoderBlock(nn.Module):
    def __init__(self,
                 num_channels_in:int,
                 num_channels_out:int,
                 kernel_size:int=3) -> None:
        super(SegNetEncoderBlock, self).__init__()
        # Encoder blocks look like
        # conv -> bn -> relu -> conv -> bn -> relu -> pool
        self.conv1 = nn.Conv2d(
            num_channels_in,
            num_channels_out,
            kernel_size = kernel_size,
            stride = 1,
            padding = 1
        )
        self.bn = nn.BatchNorm2d(num_channels_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            num_channels_out,
            num_channels_out,
            kernel_size = kernel_size,
            stride = 1,
            padding = 1
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            kernel_size = 2,
            stride = 2,
            padding = 0,
            return_indices = True
        )

    def forward(self, X:torch.Tensor) -> tuple:
        out = self.conv1(X)
        out = self.bn(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn(out)
        out = self.relu2(out)
        out, idxs = self.pool(out)

        return (out, idxs)



class SegNetDecoderBlock(nn.Module):
    def __init__(self,
                 num_channels_in:int,
                 num_channels_out:int,
                 kernel_size:int=3) -> None:
        super(SegNetDecoderBlock, self).__init__()
        # Decoder block looks like
        # unpool -> deconv -> bn -> relu -> deconv -> bn -> relu -> deconv ->
        # bn -> relu
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.deconv1 = nn.ConvTranspose2d(
            num_channels_in,
            num_channels_out,
            kernel_size = kernel_size,
            stride = 1,
            padding = 1
        )
        self.bn = nn.BatchNorm2d(num_channels_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            num_channels_out,
            num_channels_out,
            kernel_size = kernel_size,
            stride = 1,
            padding = 1
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(
            num_channels_out,
            num_channels_out,
            kernel_size = kernel_size,
            stride = 1,
            padding = 1
        )
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, X:torch.Tensor, idxs:torch.Tensor, output_size:torch.Tensor) -> torch.Tensor:
        out = self.unpool(X, idxs, output_size = output_size)
        out = self.deconv1(out)
        out = self.bn(out)
        out = self.relu1(out)
        out = self.deconv2(out)
        out = self.bn(out)
        out = self.relu2(out)
        out = self.deconv3(out)
        out = self.bn(out)
        out = self.relu3(out)

        return out


class SegNetDecoderBlockHalf(nn.Module):
    def __init__(self,
                 num_channels_in:int,
                 num_channels_out:int,
                 kernel_size:int = 3) -> None:
        super(SegNetDecoderBlockHalf, self).__init__()
        # Half Decoder block looks like
        # unpool -> deconv -> bn -> relu -> deconv
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.deconv1 = nn.ConvTranspose2d(
            num_channels_in,
            num_channels_out,
            kernel_size = kernel_size,
            stride = 1,
            padding = 1
        )
        self.bn = nn.BatchNorm2d(num_channels_out)
        self.relu = nn.ReLU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            num_channels_out,
            num_channels_out,
            kernel_size = kernel_size,
            stride = 1,
            padding = 1
        )

    def forward(self,
                X:torch.Tensor,
                idxs:torch.Tensor,
                output_size:torch.Tensor) -> torch.Tensor:
        out = self.unpool(X, idxs, output_size = output_size)
        out = self.deconv1(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.deconv2(out)

        return out



class SegNetEncoderModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.kernel_size:int = kwargs.pop('kernel_size', 3)
        super(SegNetEncoderModule, self).__init__()
        # Filter sizes for encoder = [3, 64, 128, 256, 512, 512]
        # For now, we just use the sizes from the paper

        layers = []
        fsizes = [3, 64, 128, 128, 256, 512, 512]

        for n in range(len(fsizes)):
            if n == 0:
                continue
            layers += [SegNetEncoderBlock(
                fsizes[n-1],
                fsizes[n],
                kernel_size = self.kernel_size
            )]

        self.net = nn.Sequential(*layers)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.net(X)



class SegNetDecoderModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.num_classes:int = kwargs.pop('num_classes', 12)
        self.kernel_size:int = kwargs.pop('kernel_size', 3)
        super(SegNetDecoderModule, self).__init__()
        # Deconv sizes for decoder = [512, 512, 256, 128, 64, K=num_classes]
        fsizes = [512, 512, 256, 128, 64, self.num_classes]

        layers = []
        for n in range(len(fsizes)):
            if n == 0:
                continue
            if n == len(fsizes)-1:
                layers += [SegNetDecoderBlockHalf(
                    fsizes[n],
                    fsizes[n-1],
                    kernel_size = self.kernel_size
                )]
            else:
                layers += [SegNetDecoderBlock(
                    fsizes[n],
                    fsizes[n-1],
                    kernel_size = self.kernel_size
                )]
        self.net = nn.Sequential(*layers)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        pass



