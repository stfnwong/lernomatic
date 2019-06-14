"""
ENC_DEC_ATTEN
More encoders and decoders with Attention.
Most of this is adapted from https://bastings.github.io/annotated_encoder_decoder/

Stefan Wong 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lernomatic.models import common

# debug
from pudb import set_trace; set_trace()


# ======== WRAPPER ======== #
class EncoderDecoder(common.LernomaticModel):
    def __init__(self,
                 encoder:common.LernomaticModel,
                 decoder:common.LernomaticModel,
                 generator:common.LernomaticModel,
                 src_embed:nn.Module,
                 trg_embed:nn.Module) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_embed = src_embed
        self.trg_embed = trg_embed

    def __repr__(self) -> str:
        return 'EncoderDecoder'


# ======== GENERATOR ======== #
class Generator(common.LernomaticModel):
    def __init__(self, hidden_size:int, vocab_size:int) -> None:
        self.net = GeneratorModule(hidden_size, vocab_size)
        self.import_path       : str             = 'lernomatic.model.text.enc_dec_atten'
        self.model_name        : str             = 'Generator'
        self.module_name       : str             = 'Generator'
        self.module_import_path: str             = 'lernomatic.model.text.enc_dec_atten'

    def __repr__(self) -> str:
        return 'Generator'

    def get_hidden_size(self) -> int:
        return self.net.hidden_size

    def get_vocab_size(self) -> int:
        return self.net.vocab_size


class GeneratorModule(nn.Module):
    def __init__(self, hidden_size:int, vocab_size:int) -> None:
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        super(GeneratorModule, self).__init__()
        self.proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(X), dim=-1)


# ======== ENCODER ======== #
class Encoder(common.LernomaticModel):
    def __init__(self, input_size:int, hidden_size:int, **kwargs) -> None:
        self.net = EncoderModule(input_size, hidden_size, **kwargs)
        self.import_path       : str             = 'lernomatic.model.text.enc_dec_atten'
        self.model_name        : str             = 'Encoder'
        self.module_name       : str             = 'Encoder'
        self.module_import_path: str             = 'lernomatic.model.text.enc_dec_atten'

    def __repr__(self) -> str:
        return 'Encoder'

    def forward(self,
                src:torch.Tensor,
                src_mask:torch.Tensor,
                src_lengths:list) -> tuple:
        return self.net.forward(src, src_mask, src_lengths)

    def get_input_size(self) -> int:
        return self.net.get_input_size

    def get_hidden_size(self) -> int:
        return self.net.hidden_size


class EncoderModule(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, **kwargs) -> None:
        super(EncoderModule, self).__init__()
        self.input_size  :int   = input_size
        self.hidden_size :int   = hidden_size
        self.num_layers  :int   = kwargs.pop('num_layers', 1)
        self.dropout     :float = kwargs.pop('dropout', 0.0)

        # network graph
        self.rnn = nn.GRU(
            self.input_size,
            self.hidden_size,
            dropout = self.dropout,
            batch_first = True,
            bidirectional = True,
        )

    # apply bi-directional GRU to sequence of embeddings X
    def forward(self,
                X:torch.Tensor,
                mask:torch.Tensor,
                lengths:list) -> tuple:
        """
        Arguments:
            X : torch.Tensor
                Shape (batch_size, time, dim). Input sequence of embeddings to encode
        """
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
        enc_output, final = self.rnn(packed)
        enc_output, _ = nn.utils.rnn.pad_packed_sequence(enc_output, batch_first=True)

        # manually concat the final states for both directions
        fwd_final = final[0 : final.size(0) : 2]
        bwd_final = final[1 : final.size(0) : 2]
        final = torch.cat([fwd_final, bwd_final], dim=2)    # [num_layers, batch_size, 2*dim]

        return (enc_output, final)


# ======== DECODER ======== #
class Decoder(common.LernomaticModel):
    def __init__(self, embed_size:int, hidden_size:int, **kwargs) -> None:
        self.net = DecoderModule(embed_size, hidden_size, **kwargs)
        self.import_path       : str             = 'lernomatic.model.text.enc_dec_atten'
        self.model_name        : str             = 'Decoder'
        self.module_name       : str             = 'Decoder'
        self.module_import_path: str             = 'lernomatic.model.text.enc_dec_atten'

    def __repr__(self) -> str:
        return 'Decoder'

    def forward(self,
                trg_embed:torch.Tensor,
                enc_hidden:torch.Tensor,
                enc_final:torch.Tensor,
                data_mask:torch.Tensor,
                target_mask:torch.Tensor,
                hidden=None,
                max_len:int=0) -> tuple:
        return self.net.forward(trg_embed, enc_hidden, enc_final, data_mask, target_mask, hidden, max_len)

    def get_embed_size(self) -> int:
        return self.net.embed_size

    def get_hidden_size(self) -> int:
        return self.net.hidden_size


class DecoderModule(nn.Module):
    def __init__(self, embed_size:int, hidden_size:int, **kwargs) -> None:
        super(DecoderModule, self).__init__()
        self.embed_size  :int   = embed_size
        self.hidden_size :int   = hidden_size
        self.num_layers  :int   = kwargs.pop('num_layers', 1)
        self.dropout     :float = kwargs.pop('dropout', 0.0)
        self.use_bridge  :bool  = kwargs.pop('use_bridge', True)

        # network graph
        self.rnn = nn.GRU(
            self.embed_size + 2 * self.hidden_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout = self.dropout
        )
        # bridge inits the final encoder state
        if self.use_bridge:
            self.bridge = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)
        else:
            self.bridge = None
        self.drop_layer = nn.Dropout(p = self.dropout)
        self.pre_out    = nn.Linear(
            self.hidden_size + 2 * self.hidden_size + self.embed_size,
            self.hidden_size,
            bias=False
        )

        # For now make attention fixed.
        self.attention = BahdanauAttentionModule(self.hidden_size)

    def _init_hidden_state(self, enc_final:torch.Tensor) -> torch.Tensor:
        if enc_final is None:
            return None     # start with zeros

        return torch.tanh(self.bridge(enc_final))

    def decode_step(self,
                    prev_embed:torch.Tensor,
                    enc_hidden:torch.Tensor,
                    data_mask:torch.Tensor,
                    target_mask:torch.Tensor,
                    proj_key:torch.Tensor,
                    hidden:torch.Tensor,) -> tuple:
        query = hidden[-1].unsqueeze(1)     # shape: (num_layers, B, D] -> [B, 1, D]
        context, atten_probs = self.attention(
            query=query,
            proj_key=proj_key,
            value=enc_hidden,
            mask=src_mask
        )

        # update the RNN hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.drop_layer(pre_output)
        pre_output = self.pre_out(pre_output)

        return (output, hidden, pre_output)

    def forward(self,
                trg_embed:torch.Tensor,
                enc_hidden:torch.Tensor,
                enc_final:torch.Tensor,
                data_mask:torch.Tensor,
                target_mask:torch.Tensor,
                hidden=None,
                max_len:int=0) -> tuple:

        # if we don't specify anything then unroll the entire sequence
        if max_len == 0:
            max_len = trg_mask.size(-1)

        if hidden is None:
            hidden = self._init_hidden(enc_final)

        # pre-compute projected encoder hidden states. These are like 'keys'
        # for the attention mechanism. This is only done for efficiency
        proj_key = self.attenion.compute_key(enc_hidden)

        # Cache all the intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder for the required number of steps
        for t in range(max_len):
            prev_embed = trg_embed[:, t].unsqueeze(1)
            output, hidden, pre_output = self.decode_step(
                prev_embed,
                enc_hidden,
                data_mask,
                target_mask,
                proj_key,
                hidden
            )
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)

        return (decoder_states, hiddden, pre_output_vectors)        # shape : (batch_size, N,  D)



# TODO : at some point it might be better to collect up all the attention
# models and place them in some place like lernomatic.models.attention
# ======== ATTENTION MODEL ======== #

class BahdanauAttention(common.LernomaticModel):
    def __init__(self, hidden_size:int, **kwargs) -> None:
        self.net = BahdanauAttentionModule(hidden_size, **kwargs)
        self.import_path       : str             = 'lernomatic.model.text.enc_dec_atten'
        self.model_name        : str             = 'BahdanuAttention'
        self.module_name       : str             = 'BahdanuAttention'
        self.module_import_path: str             = 'lernomatic.model.text.enc_dec_atten'

    def __repr__(self) -> str:
        return 'BahdanauAttention'

    def forward(self, query=None, proj_key=None, values=None, mask=None) -> tuple:
        return self.net.forward(query, proj_key, values, mask)

    def get_key_size(self) -> int:
        return self.net.key_size

    def get_query_size(self) -> int:
        return self.net.query_size


class BahdanauAttentionModule(nn.Module):
    def __init__(self, hidden_size:int, **kwargs) -> None:
        super(BahdanauAttentionModule, self).__init__()
        self.hidden_size :int = hidden_size
        self.key_size    :int = kwargs.pop('key_size', None)
        self.query_size  :int = kwargs.pop('query_size', None)

        if self.key_size is None:
            self.key_size = 2 * self.hidden_size

        if self.query_size is None:
            self.query_size = self.hidden_size

        # graph
        self.key_layer    = nn.Linear(self.key_size, self.hidden_size, bias=False)
        self.query_layer  = nn.Linear(self.query_size, self.hidden_size, bias=False)
        self.energy_layer = nn.Linear(self.hidden_size, 1, bias=False)
        self.alphas = None

    def compute_key(self, X:torch.Tensor) -> torch.Tensor:
        return self.key_layer(X)

    def forward(self, query=None, proj_key=None, values=None, mask=None) -> tuple:
        # project the query into the query space (the query is the decoder
        # state). The projected keys (encoder states) are already computed at
        # this point
        query = self.query_layer(query)
        # calculate scores
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        # Mask out invalid positions
        # The mask actually marks valid positions, so we invery it by ANDing
        # with zero
        scores.data.masked_fill_(mask == 0, -float('inf'))
        # Turn scores into probabilities
        alphas = F.softmax(scores, dim=1)
        # cache probs
        self.alphas = alphas

        # The context vector is just the weighted sum of the values
        context = torch.bmm(alphas, value)

        return (context, alphas)        # shapes: [B, 1, 2*D] , [B, 1, M]

