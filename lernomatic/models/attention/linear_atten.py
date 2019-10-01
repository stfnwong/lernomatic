"""
LINEAR_ATTEN
Attention network consisting of Linear/Fully-Connected Layers

Stefan Wong 2019
"""

import torch
import torch.nn as nn
from lernomatic.models import common


class AttentionNet(common.LernomaticModel):
    """
    Lernomatic model wrapper for Attention Network
    """
    def __init__(self, enc_dim: int=1, dec_dim:int = 1, atten_dim:int=1) -> None:
        self.net = AttentionNetModule(enc_dim, dec_dim, atten_dim)
        self.model_name = 'AttentionNet'
        self.module_name = 'AttentionNetModule'
        self.import_path = 'lernomatic.models.image_caption.image_caption'
        self.module_import_path = 'lernomatic.models.image_caption.image_caption'

    def __repr__(self) -> str:
        return 'AttentionNet'

    def forward(self, enc_feature, dec_hidden) -> tuple:
        return self.net(enc_feature, dec_hidden)


class AttentionNetModule(nn.Module):
    def __init__(self, enc_dim:int=1, dec_dim:int=1, atten_dim:int=1) -> None:
        """
        ATTENTION NETWORK

        Args:
            enc_dim   - size of encoded image features
            dec_dim   - size of decoder's RNN
            atten_dim - size of the attention network
        """
        super(AttentionNetModule, self).__init__()
        # save dims for __str__
        self.enc_dim   = enc_dim
        self.dec_dim   = dec_dim
        self.atten_dim = atten_dim
        self._init_network()

    def __repr__(self) -> str:
        return 'AttentionNet-%d' % self.atten_dim

    def __str__(self) -> str:
        s = []
        s.append('Attention Network\n')
        s.append('Encoder dim: %d, Decoder dim: %d, Attention dim :%d\n' %\
                 (self.enc_dim, self.dec_dim, self.atten_dim))
        return ''.join(s)

    def _init_network(self) -> None:
        self.enc_att  = nn.Linear(self.enc_dim, self.atten_dim)    # transform encoded feature
        self.dec_att  = nn.Linear(self.dec_dim, self.atten_dim)    # transform decoder output (hidden state)
        self.full_att = nn.Linear(self.atten_dim, 1)               # compute values to be softmaxed
        self.relu     = nn.ReLU()
        self.softmax  = nn.Softmax(dim=1)       # softmax to calculate weights

    def forward(self, enc_feature:torch.Tensor, dec_hidden:torch.Tensor) -> tuple:
        att1  = self.enc_att(enc_feature)        # shape : (N, num_pixels, atten_dim)
        att2  = self.dec_att(dec_hidden)         # shape : (N, atten_dim)
        att   = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)                # shape : (N, num_pixels)
        # compute the attention weighted encoding
        atten_w_enc = (enc_feature * alpha.unsqueeze(2)).sum(dim=1)     # shape : (N, enc_dim)

        return (atten_w_enc, alpha)

    def get_params(self) -> dict:
        params = {
            'enc_dim' : self.enc_dim,
            'dec_dim' : self.dec_dim,
            'atten_dim' : self.atten_dim
        }
        return params

    def set_params(self, params:dict) -> None:
        self.enc_dim   = params['enc_dim']
        self.dec_dim   = params['dec_dim']
        self.atten_dim = params['atten_dim']
        self._init_network()
