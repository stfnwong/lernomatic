"""
BEAM_SEARCH
Utilities for performing beam search

Stefan Wong 2019
"""

import torch
import heapq
from lernomatic.models import common


"""
TODO: How to implement this?

So a beam search is where you explore a graph by expanding only the most
promising nodes. We can therefore think of a 'beam' as a path through this
graph.

"""


class BeamSearcher(object):
    def __init__(self,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticMode=None,
                 atten_net:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.encoder   = encoder        # TODO : may not need this
        self.decoder   = decoder
        self.atten_net = atten_net

        # keyword args
        self.batch_size:int = kwargs.pop('batch_size', 64)
        self.beam_size:int  = kwargs.pop('beam_size', 3)    # number of beams

        # device
        self.device_id : int = kwargs.pop('device_id', -1)

        # Send everything to the device
        self._init_device()
        self._send_to_device()

        # init scores
        self.topk_scores = torch.empty(
            (self.batch_size, self.beam_size),
            dtype=torch.float
        )
        self.topk_ids = torch.empty(
            (self.batch_size, self.beam_size),
            dtype=torch.float
        )
        self.best_scores = torch.full(
            [self.batch_size],
            -1e10,
            dtype=torch.float,
        )
        self.best_scores = self.best_scores.to(self.device)


    def __repr__(self) -> str:
        return 'BeamSearcher'

    # It might make more sense for this to be the sequence length
    def __len__(self) -> int:
        return self.beam_size

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

    # Do one 'step' of beam search
    # Note that for the decoder in
    # lernomatic.models.image_caption.image_caption
    #
    # Input :
    #  enc_feature, enc_capt, capt_lengths
    #
    # Returns:
    # predictions, enc_capt, decode_lengths, alphas, sort_ind
    #
    # TODO : Should we return a dict (and therefore they can have named keys)
    def advance(self, log_probs:torch.Tensor, atten:torch.Tensor) -> None:
        pass

    # TODO : data needs to come from top-level inference file
    def eval(self,
             feature:torch.Tensor,
             enc_capt:torch.Tensor,
             capt_lengths:torch.Tensor) -> torch.Tensor:

        if self.encoder is not None:
            self.encoder.set_eval()

        self.decoder.set_eval()

        # move data to device
        feature = feature.to(self.device)
        enc_capt = enc_capt.to(self.device)
        capt_lengths = capt_lengths.to(self.device)

        # Set up scores for evaluation

        hypothesis = [[] for _ in range(self.batch_size)]

        if self.atten_net is not None:
            atten_net_input = self.decoder.get_hidden_state()

        for step in range(max_length):
            enc_feature = self.encoder.forward(feature)
            embed_vector = self.decoder.embedding(enc_feature)
            log_probs, atten = self.decoder.forward()

            self.advance(log_probs, atten)





# TODO : support loading from Trainer checkpoint
