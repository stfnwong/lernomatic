"""
IM2LATEX
Decoder for im2latex

TODO : Move OCRNet into here as Im2LatexEncoder?

Stefan Wong 2019
"""

import math
import torch
import torch.nn as nn
from lernomatic.models import common

# This number comes from "Attention Is All You Need"
POS_DIV_CONSTANT = 10000.0

# ======== POSITIONAL ENCODER ======== #
# This is more or less taken from the Pytorch documentation
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoder(nn.Module):
    def __init__(self, inp_size:int, dropout:float=0.1, max_len:int=5000) -> None:
        super(PositionalEncoder, self).__init__()

        # "graph" of encoder
        self.dropout = nn.Dropout(p = dropout)
        pos_enc = torch.zeros(max_len, inp_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, inp_size, 2).float() * (-math.log(POS_DIV_CONSTANT) / inp_size))

        # Create the encoding. In this case the encoding is a geometric
        # progression from 2pi -> 10000 - 2pi. The theory is here is that the
        # model can learn to attend to relative positions in the input vector,
        # since for any fixed offset k, the position P Epos + k can be
        # represented as a linear function of P Epos.
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pos_enc', pos_enc)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = X + self.pos_enc[: X.size(0), :]
        return self.dropout(out)


# ======== ENCODER SIDE ======== #
class Im2LatexEncoder(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = Im2LatexEncoderModule(**kwargs)
        self.import_path       : str             = 'lernomatic.models.text'
        self.model_name        : str             = 'Im2LatexEncoder'
        self.module_name       : str             = 'Im2LatexEncoderModule'
        self.module_import_path: str             = 'lernomatic.models.text'

    def __repr__(self) -> str:
        return 'Im2LatexEncoder'

    def get_params(self) -> dict:
        return {
            'kernel_size'        : self.net.kernel_size,
            'num_input_channels' : self.net.num_input_channels,
            'enc_out_dim'        : self.net.enc_out_dim
        }



class Im2LatexEncoderModule(nn.Module):
    """
    Im2LatexEncoderModule
    CNN Encoder for Im2Latex challenge
    """
    def __init__(self, **kwargs) -> None:
        self.kernel_size:int        = kwargs.pop('kernel_size', 3)
        self.num_input_channels:int = kwargs.pop('num_input_channels', 3)
        self.enc_out_dim:int        = kwargs.pop('enc_out_dim', 512)

        super(Im2LatexEncoderModule, self).__init__()

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


# ======== DECODER SIDE ======== #
class Im2LatexDecoder(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = Im2LatexDecoderModule(**kwargs)
        self.import_path       : str             = 'lernomatic.models.text'
        self.model_name        : str             = 'Im2LatexDecoder'
        self.module_name       : str             = 'Im2LatexDecoderModule'
        self.module_import_path: str             = 'lernomatic.models.text'

    def __repr__(self) -> str:
        return 'Im2LatexDecoder'

    def get_params(self) -> dict:
        return {
            'inp_size'    : self.net.inp_size,
            'embed_size'  : self.net.embed_size,
            'hidden_size' : self.net.hidden_size
        }

class Im2LatexDecoderModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.inp_size:int    = kwargs.pop('inp_size', 512)
        self.embed_size:int  = kwargs.pop('embed_size', 512)
        self.hidden_size:int = kwargs.pop('hidden_size', 512)

        super(Im2LatexDecoderModule, self).__init__()

        # Network graph
        self.embedding = nn.Embedding(self.inp_size, self.embed_size)
        self.rnn       = nn.LSTMCell(
            self.inp_size + self.hidden_size,
            self.hidden_size
        )

    def init_decoder(self, enc_feature:torch.Tensor) -> tuple:
        pass


    def forward(self, enc_image:torch.Tensor, formula:torch.Tensor) -> torch.Tensor:
        pass



    #def forward_step(self, e
