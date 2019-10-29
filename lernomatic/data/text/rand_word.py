"""
RAND_WORD
Random word generation

Stefan Wong 2019
"""

import torch
import numpy as np

# custom batch object for this trainer
class RandBatch(object):
    def __init__(self, src:tuple, trg:tuple, pad_index:int=0) -> None:
        src, src_lengths = src

        self.src         = src
        self.src_lengths = src_lengths
        self.src_mask    = (src != pad_index).unsqueeze(-2)
        self.nseqs       = src.size(0)

        self.trg         = None
        self.trg_y       = None
        self.trg_mask    = None
        self.trg_legnths = None
        self.num_tokens  = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

    def __repr__(self) -> str:
        return 'RandBatch'

    def __str__(self) -> str:
        s = []
        s.append('RandBatch \n')
        s.append('src shape : %s\n' % str(self.src.shape))
        s.append('src lengths : %s\n' % str(self.src_lengths))

        if self.trg is not None:
            s.append('trg shape : %s\n' % str(self.trg.shape))
            s.append('trg lengths : %s\n' % str(self.trg_lengths))
            s.append('trg_y shape : %s\n' % str(self.trg_y.shape))

        return ''.join(s)

    #def __len__(self) -> int:
    #    return self.data.size(0)

    def to(self, device:torch.device) -> None:

        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)
        if self.trg is not None:
            self.trg = self.trg.to(device)
            self.trg_y = self.trg_y.to(device)
            self.trg_mask = self.trg_mask.to(device)


# generator for synthetic translation task
def data_gen(num_words:int = 11,
             batch_size:int = 16,
             num_batches:int = 100,
             length:int = 10,
             pad_index:int = 0,
             sos_index:int = 1,
             device=torch.device('cpu')) -> RandBatch:
    for i in range(num_batches):
        data = torch.from_numpy(
            np.random.randint(1, num_words, size=(batch_size, length))
        )
        data[:, 0] = sos_index
        data = data.to(device)
        src = data[:, 1:]
        trg = data
        src_lengths = [length-1] * batch_size
        trg_lengths = [length] * batch_size

        # note: data needs to have shape (batch_size, time, dim)

        yield RandBatch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)


# A sort of hack dataloader
class RandWordBatchGenerator(object):
    def __init__(self, **kwargs) -> None:
        self.num_words   :int = kwargs.pop('num_words', 11)
        self.batch_size  :int = kwargs.pop('batch_size', 16)
        self.embed_dim   :int = kwargs.pop('embed_dim', 64)
        self.length      :int = kwargs.pop('length', 10)
        self.pad_index   :int = kwargs.pop('pad_index', 0)
        self.sos_index   :int = kwargs.pop('sos_index', 1)
        self.num_batches :int = kwargs.pop('num_batches', 64)

    def __len__(self) -> int:
        return self.num_batches

    def __repr__(self) -> str:
        return 'RandWordBatchGenerator'

    def __getitem__(self, idx:int) -> RandBatch:
        if idx > len(self):
            raise StopIteration

        data = torch.from_numpy(
            np.random.randint(1, self.num_words, size=(self.batch_size, self.length))
            #np.random.randint(1, self.num_words, size=(self.batch_size, self.length, self.num_words))
            #np.random.randint(1, self.num_words, size=(self.batch_size, self.length, self.embed_dim))
        )
        data[:, 0] = self.sos_index
        src = data[:, 1:]
        trg = data
        src_lengths = [self.length-1] * self.batch_size
        trg_lengths = [self.length] * self.batch_size

        return RandBatch((src, src_lengths), (trg, trg_lengths), pad_index=self.pad_index)

