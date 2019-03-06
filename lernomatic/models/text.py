"""
TEXT
Models for working with text

Stefan Wong 2019
"""

import torch.nn as nn
from torch.autograd import Variable


class TextRNN(nn.Module):
    def __init__(self, num_tokens, **kwargs):
        self.input_dim  = kwargs.pop('input_dim', 200)
        self.hidden_dim = kwargs.pop('hidden_dim', 200)
        self.num_layers = kwargs.pop('num_layers', 2)
        self.rnn_type   = kwargs.pop('rnn_type', 'RNN_TANH')
        super(TextRNN, self).__init__()

        # internals
        self.encoder = nn.Embedding(num_tokens, self.input_dim)
        self.rnn     = getattr(nn, self.rnn_type)(
            self.input_dim,
            self.hidden_dim,
            self.num_layers,
            bias=False
        )
        self.decoder = nn.Linear(self.hidden_dim, num_tokens)
        self._init_weights()


    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters.data())
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers, bsz, self.num_hidden).zero_()),
                    Variable(weight.new(self.num_layers, bsz, self.num_hidden).zero_())
            )
        else:
            return Variable(weight.new(self.num_layers, bsz, self.num_hidden).zero_())

    def forward(self, X, hidden):
        emb = self.encoder(X)
        output, hidden = self.rnn(emb, hidden)
        decode = self.decoder(
            output.view(
                output.size(0) * output.size(1),
                output.size(2)
            )
        )

        return decode.view(output.size(0), output.size(1), decode.size(1)), hidden
