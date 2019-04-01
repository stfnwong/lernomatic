"""
COCO
Some other modules for a COCO experiment

Stefan Wong 2019
"""

import torch
import torchvision
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from lernomatic.models import common
# caption utils
from lernomatic.util import caption as caption_utils


class COCOEncoder(common.LernomaticModel):
    def __init__(self, encoder: common.LernomaticModel, vocab_size: int, **kwargs) -> None:
        self.net = COCOEncoderModule(encoder, vocab_size, **kwargs)
        self.model_name = 'COCOEncoder'
        self.module_name = 'COCOEncoderModule'
        self.import_path = 'lernomatic.models.image_caption'
        self.module_import_path = 'lernomatic.models.image_caption'

    def __repr__(self) -> str:
        return 'COCOEncoder'

class COCOEncoderModule(nn.Module):
    def __init__(self, encoder, vocab_size, **kwargs) -> None:
        self.embed_size     = kwargs.pop('embed_size', 255)
        self.rnn_size       = kwargs.pop('rnn_size', 256)
        self.num_rnn_layers = kwargs.pop('num_rnn_layers', 2)
        self.share_weights  = kwargs.pop('share_weights', False)
        super(COCOEncoderModule, self).__init__()

        # components
        self.encoder = encoder
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, self.embed_size)
        self.rnn = nn.LSTM(self.embed_size, self.rnn_size, num_layers=self.num_rnn_layers)
        self.classifier = nn.Linear(self.rnn_size, vocab_size)
        self.embedder = nn.Embedding(vocab_size, self.embed_size)

        if self.share_weights:
            self.embedder.weights = self.classifier.weights

    def forward(self, imgs, captions, caplens) -> tuple:
        embeddings        = self.embedder(captions)
        enc_img           = self.encoder(imgs).unsqueeze(0)
        embeddings        = torch.cat([enc_img, embeddings], 0)
        packed_embeddings = pack_padded_sequence(embeddings, caplens)
        features, state   = self.rnn(packed_embeddings)
        pred              = self.classifier(features[0])

        return pred, state


    def generate(self, img, scale_size=256, crop_size=224):
        pass


