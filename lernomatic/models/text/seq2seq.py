"""
SEQ2SEQ
Some seq2seq models for Text

Stefan Wong 2019
"""

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from lernomatic.models import common

# debug
#from pudb import set_trace; set_trace()


# TODO : move attention networks to sub-module
class EncoderRNN(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = EncoderRNNModule(**kwargs)
        self.import_path = 'lernomatic.models.text.seq2seq'
        self.model_name = 'EncoderRNN'
        self.module_name = 'EncoderRNNModule'
        self.module_import_path = 'lernomatic.models.text.seq2seq'

    def __repr__(self) -> str:
        return 'EncoderRNN'

    def forward(self,
                input_seq:torch.Tensor,
                input_lengths:torch.Tensor,
                hidden=None) -> torch.Tensor:
        return self.net.forward(input_seq, input_lengths, hidden)

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_num_layers(self) -> int:
        return self.net.num_layers

    def get_params(self) -> dict:
        params = super(EncoderRNN, self).get_params()
        params['init_params'] = {
            'hidden_size': self.net.hidden_size,
            'num_layers' : self.net.num_layers,
            'num_words'  : self.net.num_words,
            'dropout'    : self.net.dropout,
        }

        return params

    def set_params(self, params:dict) -> None:
        # regular model stuff
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)
        self.net = mod(
            hidden_size = params['init_params']['hidden_size'],
            num_layers  = params['init_params']['num_layers'],
            num_words   = params['init_params']['num_words'],
            dropout     = params['init_params']['dropout'],
        )
        self.net.load_state_dict(params['model_state_dict'])


# Produce encoded context vectors
class EncoderRNNModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(EncoderRNNModule, self).__init__()
        self.hidden_size :int       = kwargs.pop('hidden_size', 512)
        self.num_layers  :int       = kwargs.pop('num_layers', 1)
        self.num_words   :int       = kwargs.pop('num_words', 1000)
        self.dropout     :float     = kwargs.pop('dropout', 0.0)
        self.embedding   :nn.Module = kwargs.pop('embedding', None)

        if self.embedding is None:
            self.embedding = nn.Embedding(self.num_words, self.hidden_size)
        if self.num_layers == 1:
            self.dropout = 0.0

        self.rnn = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True
        )

    def forward(self,
                input_seq:torch.Tensor,
                input_lengths:torch.Tensor,
                hidden=None) -> torch.Tensor:
        embed = self.embedding(input_seq)
        packed = nn.utils.rnn.pack_padded_sequence(embed, input_lengths)
        outputs, hidden = self.rnn(packed, hidden)
        # unpack the packed padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # sum bidirectional GRU outputs
        outputs = outputs[:, :, 0:self.hidden_size] + outputs[:, :, self.hidden_size:]

        # TODO :  hidden should be 3 dimensions here...
        return (outputs, hidden)


# Calculates attention weightings over all encoder hidden states
class GlobalAttentionNet(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = GlobalAttentionNetModule(hidden_size, **kwargs)
        self.import_path        = 'lernomatic.models.text.seq2seq'
        self.model_name         = 'GlobalAttentionNet'
        self.module_name        = 'GlobalAttentionNetModule'
        self.module_import_path = 'lernomatic.models.text.seq2seq'

    def __repr__(self) -> str:
        return 'GlobalAttentionNet'

    def forward(self,
                hidden:torch.Tensor,
                enc_output:torch.Tensor) -> torch.Tensor:
        return self.net.forward(hidden, enc_output)

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_score_method(self) -> str:
        return self.net.score_method

    def get_params(self) -> dict:
        params = super(GlobalAttentionNet, self).get_params()
        params['init_params'] = {
            'hidden_size' : self.net.hidden_size,
            'score_method' : self.net.score_method,
        }

        return params

    def set_params(self, params:dict) -> None:
        # regular model stuff
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)
        self.net = mod(
            hidden_size  = params['init_params']['hidden_size'],
            score_method = params['init_params']['score_method'],
        )
        self.net.load_state_dict(params['model_state_dict'])


class GlobalAttentionNetModule(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(GlobalAttentionNetModule, self).__init__()
        self.hidden_size:int  = kwargs.pop('hidden_size', 512)
        self.score_method:str = kwargs.pop('score_method', 'dot')

        # check score_method
        valid_score_methods = ['dot', 'general', 'concat']
        if self.score_method not in valid_score_methods:
            raise ValueError('Invalid score_method [%s], must be one of %s' %\
                             (str(self.score_method), str(valid_score_methods))
            )

        if self.score_method == 'general':
            self.atten = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.score_method == 'concat':
            self.atten = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self,
                  hidden:torch.Tensor,
                  enc_output:torch.Tensor) -> torch.Tensor:
        return torch.sum(hidden * enc_output, dim=2)

    def general_score(self,
                      hidden:torch.Tensor,
                      enc_output:torch.Tensor) -> torch.Tensor:
        energy = self.atten(enc_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self,
                     hidden:torch.Tensor,
                     enc_output:torch.Tensor) -> torch.Tensor:
        energy = self.atten(
            torch.cat(
                hidden.expand(enc_output.size(0), -1),
                -1,
                enc_output)
        ).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self,
                hidden:torch.Tensor,
                enc_output:torch.Tensor) -> torch.Tensor:
        # compute attention weights
        if self.score_method == 'dot':
            atten_w = self.dot_score(hidden, enc_output)
        elif self.score_method == 'general':
            atten_w = self.general_score(hidden, enc_output)
        elif self.score_method == 'concat':
            atten_w = self.concat_score(hidden, enc_output)

        # transpose batch dimension
        atten_w = atten_w.t()

        return F.softmax(atten_w, dim=1).unsqueeze(1)



#  Decoder module
class LuongAttenDecoderRNN(common.LernomaticModel):
    def __init__(self,
                 hidden_size:int,
                 output_size:int,
                 **kwargs) -> None:
        self.net = LuongAttenDecoderRNNModule(
            hidden_size,
            output_size,
            **kwargs)
        self.import_path        = 'lernomatic.models.text.seq2seq'
        self.model_name         = 'LuongAttenDecoderRNN'
        self.module_name        = 'LuongAttenDecoderRNNModule'
        self.module_import_path = 'lernomatic.models.text.seq2seq'

    def __repr__(self) -> str:
        return 'LuongAttenDecoderRNN'

    def get_num_layers(self) -> int:
        return self.net.num_layers

    def forward(self,
                input_step:torch.Tensor,
                prev_hidden:torch.Tensor,
                enc_out:torch.Tensor) -> torch.Tensor:
        return self.net.forward(input_step, prev_hidden, enc_out)

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_output_size(self) -> int:
        return self.net.output_size

    def get_score_method(self) -> str:
        return self.net.get_score_method()

    def get_params(self) -> dict:
        params = super(LuongAttenDecoderRNN, self).get_params()
        params['init_params'] = {
            'hidden_size'  : self.net.hidden_size,
            'output_size'  : self.net.output_size,
            'embedding'    : self.net.embedding,
            'num_layers'   : self.net.num_layers,
            'dropout'      : self.net.dropout,
            'score_method' : self.net.score_method
        }

        return params

    def set_params(self, params:dict) -> None:
        # regular model stuff
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)
        # TODO : check that the state dict restores the embedding params
        # correctly
        self.net = mod(
            params['init_params']['hidden_size'],
            params['init_params']['output_size'],
            embedding = params['init_params']['embedding'],
            num_layers = params['init_params']['num_layers'],
            dropout = params['init_params']['dropout'],
            score_method = params['init_params']['score_method'],
        )
        self.net.load_state_dict(params['model_state_dict'])


"""
Method:

    1) Get embedding of current word
    2) Pass embedded word forward through unidirectional GRU
    3) Compute attention weights based on GRU output
    4) Multiply attention weights by encoder outputs to get
    new attention-weighted context vector
    5) Concat the weighted context vector and the GRU output
    according to score method
    6) Predict next word
    7) Return output and final hidden state

"""
# TODO: factor attention out to new sub-module?
class LuongAttenDecoderRNNModule(nn.Module):
    def __init__(self,
                 hidden_size:int,
                 output_size:int,
                 **kwargs) -> None:
        super(LuongAttenDecoderRNNModule, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # keyword args
        self.embedding    :nn.Module = kwargs.pop('embedding', None)
        self.num_layers   :int       = kwargs.pop('num_layers', 1)
        self.dropout      :float     = kwargs.pop('dropout', 0.0)
        self.score_method :str       = kwargs.pop('score_method', 'dot')

        if self.embedding is None:
            self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # setup network
        if self.num_layers == 1:
            self.dropout = 0.0
        self.embed_dropout = nn.Dropout(self.dropout)
        self.rnn = nn.GRU(
            self.hidden_size,
            self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        self.concat = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.out    = nn.Linear(self.hidden_size, self.output_size)
        self.atten  = GlobalAttentionNetModule(
            hidden_size = self.hidden_size,
            score_method=self.score_method
        )

    def forward(self,
                input_step:torch.Tensor,
                prev_hidden:torch.Tensor,
                enc_out:torch.Tensor) -> torch.Tensor:
        embed = self.embedding(input_step)
        embed = self.embed_dropout(embed)       # <- dim() here = 3
        # pass embedded vector through GRU
        gru_out, gru_hidden = self.rnn(embed, prev_hidden)
        # compute attention weights
        atten_w = self.atten(gru_out, enc_out)
        # context vector is batch-matrix-matrix product of attention weights
        # and encoder output
        context_vec = atten_w.bmm(enc_out.transpose(0, 1))

        gru_out     = gru_out.squeeze(0)
        context_vec = context_vec.squeeze(1)
        concat_inp  = torch.cat((gru_out, context_vec), 1)
        concat_out  = torch.tanh(self.concat(concat_inp))

        # predict next word
        output = self.out(concat_out)
        output = F.softmax(output, dim=1)

        return (output, gru_hidden)
