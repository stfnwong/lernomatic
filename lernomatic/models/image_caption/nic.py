"""
NIC
Models for image captioning based on the 'Neural Image Caption' model

Stefan Wong 2019
"""

import torch
import torchvision
import importlib
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from lernomatic.models import common
# caption utils
from lernomatic.util import caption as caption_utils


# ==== SIMPLIFIED DECDOER ==== #

class NICDecoder(common.LernomaticModel):
    def __init__(self,
                 embed_dim: int = 512,
                 dec_dim: int = 512,
                 vocab_size: int = 1,
                 enc_dim: int = 2048,
                 dropout: float = 0.5,
                 **kwargs) -> None:
        self.net = NICDecoderModule(
            embed_dim, dec_dim, vocab_size,
            enc_dim, dropout, **kwargs)
        self.model_name         = 'NICDecoder'
        self.module_name        = 'NICDecoderModule'
        self.import_path        = 'lernomatic.models.image_caption'
        self.module_import_path = 'lernomatic.models.image_caption'

    def __repr__(self) -> str:
        return 'NICDecoder'

    def __str__(self) -> str:
        return 'NICDecoder-%d' % self.net.vocab_size


class NICDecoderModule(nn.Module):
    def __init__(self,
                 dec_dim=1
                 , vocab_size=1,
                 enc_dim=2048,
                 dropout=0.5,
                 **kwargs) -> None:
        """
        Simplified LSTM Decoder for image captioning

        Args:
            embed_dim  - size of embedding layer
            dec_dim    - size of decoder RNN
            vocab_size - size of the vocabulary
            enc_dim    - size of encoded features
            dropout    - the dropout ratio
        """
        super(NICDecoderModule, self).__init__()
        # copy params
        self.enc_dim    = enc_dim
        self.dec_dim    = dec_dim
        self.embed_dim  = embed_dim
        self.vocab_size = vocab_size
        self.dropout    = dropout
        self.device     = None      # archive the device for some internal forward pass stuff
        # create the actual network
        self._init_network()
        self.init_weights()

    def _init_network(self) -> None:
        # Create internal layers
        self.embedding   = nn.Embedding(self.vocab_size, self.embed_dim)
        self.drop        = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.enc_dim, self.dec_dim, bias=True)     # decoding LSTM cell
        # linear layer to find initial hidden state of LSTM
        self.init_h      = nn.Linear(self.enc_dim, self.dec_dim)
        # linear layer to find initial cell state of LSTM
        self.init_c      = nn.Linear(self.enc_dim, self.dec_dim)
        # linear layer to create sigmoid activated gate
        self.f_beta      = nn.Linear(self.dec_dim, self.enc_dim)
        self.sigmoid     = nn.Sigmoid()
        # linear layer to find scores over vocab
        self.fc          = nn.Linear(self.dec_dim, self.vocab_size)


    def _inti


    def forward(self, enc_feature, enc_capt, capt_lengths) -> tuple:
        """
        FOWARD PASS

        Args:
            enc_feature  - Tensor of dimension (N, enc_image_size, enc_image_size, enc_dim)
            enc_capt     - Tensor of dimension (N, max_capt_length)
            capt_lengths - Tensor of dimension (N, 1)

        Return:
            predictions, enc_capt, decode_lengths, alphas, sort_ind?
        """
        N          = enc_feature.size(0)             # batch size
        enc_dim    = enc_feature.size(-1)
        vocab_size = self.vocab_size

        # flatten image
        enc_feature = enc_feature.view(N, -1, enc_dim)  # aka : (N, num_pixels, enc_dim)
        num_pixels  = enc_feature.size(1)

        # pack_padded_sequence expects a sorted tensor (ie: longest to shortest)
        capt_lengths, sort_ind = capt_lengths.squeeze(1).sort(dim=0, descending=True)
        enc_feature = enc_feature[sort_ind]
        enc_capt    = enc_capt[sort_ind]

        # Embeddings
        embeddings = self.embedding(enc_capt)       # shape = (N, max_capt_len, embed_dim)
        # init LSTM state
        h, c = self.init_hidden_state(enc_feature)  # shape = (N, dec_dim)
        # we won't decode the <end> position, since we've finished generating
        # as soon as we generate <end>. Therefore decoding lengths are
        # actual_lengths-1
        decode_lengths = (capt_lengths - 1).tolist()

        # Create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(N, max(decode_lengths), vocab_size).to(self.device)
        alphas      = torch.zeros(N, max(decode_lengths), num_pixels).to(self.device)

        # sample...
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))   # gating scalar (batch_size_t, enc_dim)
            atten_w_enc = atten_w_enc * gate
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], atten_w_enc], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )   # shape is (batch_size_t, vocab_size)

            preds = self.fc(self.drop(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, ]       = alpha


        pass
