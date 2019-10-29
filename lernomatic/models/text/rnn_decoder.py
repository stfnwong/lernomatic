"""
RNN_DECODER
Generic RNN Decoder for text networks

Stefan Wong 2019
"""

import torch
import torch.nn as nn
from lernomatic.models import common



class TextRNNDecoder(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net               : torch.nn.Module = TextRNNDecoderModule(**kwargs)
        self.import_path       : str             = 'lernomatic.models.text'
        self.model_name        : str             = 'TextRNNDecoder'
        self.module_name       : str             = 'TextRNNDecoderModule'
        self.module_import_path: str             = 'lernomatic.models.text'

    def __repr__(self) -> str:
        return 'TextRNNDecoder'

    def get_params(self) -> dict:
        return {
            'inp_size'    : self.net.inp_size,
            'embed_size'  : self.net.embed_size,
            'hidden_size' : self.net.hidden_size
        }

    def set_params(self, params:dict) -> None:
        pass



class TextRNNDecoderModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        self.inp_size:int    = kwargs.pop('inp_size', 512)
        self.embed_size:int  = kwargs.pop('embed_size', 512)
        self.hidden_size:int = kwargs.pop('hidden_size', 512)
        super(TextRNNDecoderModule, self).__init__()


        # Create network graph
        self.embedding = nn.Embedding(self.inp_size, self.embed_size)
        self.rnn       = nn.LSTMCell(
            self.hidden_size + self.embed_size,
            self.hidden_size
        )

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        out = self.embedding(X)

