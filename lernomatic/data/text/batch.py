"""
BATCH
Utils for transforming text data to batch form

Stefan Wong 2019
"""

from typing import Tuple


import itertools
import torch
from lernomatic.data.text import vocab


def idxs_from_setence(voc:vocab.Vocabulary,
                      sentence:str,
                      seperator:str=' ',
                      eos_token=None) -> list:
    if eos_token is None:
        eos_token = voc.eos_token
    return [voc.word2idx[word] for word in sentence.split(seperator)] + eos_token


def zero_padding(l, fill_val:str) -> list:
    return list(itertools.zip_longest(*l, fillvalue=fill_val))


def binary_matrix(l:str, value:str) -> list:
    m = []
    for n, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)

    return m


def pad_input_seq(l:str, voc:vocab.Vocabulary) -> Tuple[torch.Tensor, list]:
    batch_idxs = [idxs_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.Tensor([len(idx) for idx in batch_idxs])
    pad_list = zero_padding(batch_idxs)
    pad_var = torch.LongTensor(pad_list)

    return (pad_var, lengths)


def pad_output_seq(l, voc:vocab.Vocabulary) -> Tuple[torch.Tensor, torch.Tensor, list]:
    batch_idxs = [idxs_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(idx) for idx in batch_idxs])
    pad_list = zero_padding(batch_idxs)

    mask = binary_matrix(pad_list)
    mask = torch.ByteTensor(mask)

    padvar = torch.LongTensor(pad_list)

    return (padvar, mask, max_target_len)


def batch_convert(voc:vocab.Vocabulary,
                  pair_batch,
                  seperator:str=' ') -> tuple:
    pair_batch.sort(key=lambda x: len(x[0].split(seperator)), reverse=True)
    inp_batch = []
    out_batch = []

    for pair in pair_batch:
        inp_batch.append(pair[0])
        out_batch.append(pair[1])

    inp_data, lengths = pad_input_seq(inp_batch, voc)
    out_data, mask, max_target_len = pad_output_seq(out_batch, voc)

    return (inp_data, lengths, out_data, mask, max_target_len)
