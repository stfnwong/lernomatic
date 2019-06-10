"""
GREEDY_DECODER
A greedy search decoder for evaluating text data

Stefan Wong 2019
"""

from typing import Tuple
import torch
from lernomatic.model import common


# TODO: I've set the models to None here since I think I want to implement checkpoint loading
class GreedySearchDecoder(object):
    def __init__(self,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.encoder = encoder
        self.decoder = decoder

        # handle keywords
        self.verbose    = kwargs.pop('verbose', False)
        self.max_length = kwargs.pop('max_length', 30)
        self.sos_token  = kwargs.pop('sos_token', 1)
        self.device_id  = kwargs.pop('device_id', -1)

        self._init_device()
        self._send_to_device()

    def __repr__(self) -> str:
        return 'GreedySearchDecoder'

    def _init_device(self) -> None:
        if self.device_id < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % self.device_id)

    def _send_to_device(self) -> None:
        if self.encoder is not None:
            self.encoder.send_to(self.device)
        if self.decoder is not None:
            self.decoder.send_to(self.device)

    def search(self,
               input_seq:torch.Tensor,
               input_lengths:list,
               max_length=0) -> Tuple[torch.Tensor, torch.Tensor]
        if max_length == 0:
            max_length = self.max_length

        # encode sequence, setting final hidden state of encoder to be initial
        # hidden state of decoder.
        enc_output, enc_hidden = self.encoder(input_seq, input_lengths)
        dec_hidden = enc_hidden[0 : self.decoder.get_num_layers()]
        # init the decoder input to the sos_token. Default is 1, but it should
        # be set to whatever the value in the vocab is at init time
        dec_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * self.sos_token

        # init some tensors to hold decoded words
        out_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        out_scores = torch.zeros([0], device=self.device)

        for t in range(max_length):
            dec_output, dec_hidden = self.decoder(dec_input, dec_hidden)
            dec_scores, dec_input = torch.max(dec_output, dim=1)

            out_tokens = torch.cat((out_tokens, dec_input), dim=0)
            out_scores = torch.cat((out_scores, dec_scores), dim=0)

            dec_input = torch.unsqueeze(dec_input, 0)

        return (out_tokens, out_scores)

    # TODO: loading and saving from checkpoint data
