"""
CHAR_RNN
Character RNN model

Stefan Wong 2019
"""

import torch
import torch.nn as nn
from lernomatic.models import common


def detach_history(h):
    if type(h) is torch.Tensor:
        return h.detach()
    else:
        return tuple(detach_history(v) for v in h)


class CharSeqStateful(common.LernomaticModel):
    def __init__(self, vocab_size:int, hidden_size:int, **kwargs) -> None:
        self.net = CharSeqStatefulModule(
            vocab_size,
            hidden_size,
            **kwargs)
        self.import_path = 'lernomatic.models.text.char_rnn'
        self.model_name = 'CharSeqStateful'
        self.module_name = 'CharSeqStatefulModule'
        self.module_import_path = 'lernomatic.models.text.char_rnn'

    def __repr__(self) -> str:
        return 'CharSeqStateful'


class CharSeqStatefulModule(nn.Module):
    def __init__(self, vocab_size:int, hidden_size:int, **kwargs) -> None:
        super(CharSeqStatefulModule, self).__init__()
        self.vocab_size:int  = vocab_size
        self.hidden_size:int = hidden_size
        self.num_fac:int     = kwargs.pop('num_fac', 42)        # TODO : clear up meaning of this param
        self.num_layers:int  = kwargs.pop('num_layers', 1)
        self.dropout:float   = kwargs.pop('dropout', 0.0)
        batch_size:int  = kwargs.pop('batch_size', 64)

        # network graph
        self.embed = nn.Embedding(self.vocab_size, self.num_fac)
        self.rnn = nn.LSTM(self.num_fac,
                           self.hidden_size,
                           self.num_layers,
                           dropout=self.dropout)
        self.fc = nn.Linear(self.num_hidden, self.vocab_size)
        self._init_hidden_state(batch_size)

    def _init_hidden_state(self, batch_size:int) -> None:
        self.hidden_state = (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                             torch.zeros(self.num_layers, batch_size, self.hidden_size)
                             )

    def forward(self, char_sequence:torch.Tensor) -> torch.Tensor:
        batch_size = char_sequence[0].size(0)
        if self.hidden_state[0].size(1) != batch_size:
            self._init_hidden_state(batch_size)

        embedding = self.embed(char_sequence)
        output, hidden = self.rnn(embedding, self.hidden_state)
        self.hidden_state = detach_history(hidden)

        return F.log_softmax(self.fc(output), dim=1).view(-1, self.vocab_size)
