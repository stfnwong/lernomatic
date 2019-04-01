"""
VAL_CAPTION
Tools for validating captions

Stefan Wong 2019
"""

import torch
from nltk.translate.bleu_score import corpus_bleu

from lernomatic.models import common
from lernomatic.data.text import word_map




def caption_beam_search(
    encoder:common.LernomaticModel,
    decoder:common.LernomaticModel,
    wmap : word_map.WordMap,
    image : torch.Tensor,
    beam_size:int) -> tuple:


    encoder.set_eval()
    decoder.set_eval()



    while True:
        embeddings = deocder.embedding(k_prev_words).squeeze(1) # (s, embed_dim)
        atten_weight_emb, _ = decoder.attention(encoder_out, h)
