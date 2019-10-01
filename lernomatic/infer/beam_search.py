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

The beam width is the number of beams that are 'in-flight' at one time.
"""

class Beam(object):
    def __init__(self) -> None:
        self.beam_size:int = 0

    def __repr__(self) -> str:
        return 'Beam'

    def __len__(self) -> int:
        return self.beam_size



class BeamSearcher(object):
    def __init__(self,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticMode=None,
                 **kwargs) -> None:
        self.encoder = encoder
        self.decoder = decoder

        # keyword args
        self.batch_size:int = kwargs.pop('batch_size', 64)
        self.beam_size:int = kwargs.pop('beam_size', 3)

        # scores
        self.topk_scores = torch.empty(
            (self.batch_size, self.beam_size),
            dtype=torch.float
        )

    def __repr__(self) -> str:
        return 'BeamSearcher'

    def __len__(self) -> int:
        return self.beam_size


    # Do one 'step' of beam search
    def advance(self, log_probs:torch.Tensor, atten:torch.Tensor) -> None:
        pass


    # TODO : a good question here is - where does the data come from?
    def eval(self, data:torch.Tensor, label:torch.Tensor) -> None:
        pass


# TODO : support loading from Trainer checkpoint
