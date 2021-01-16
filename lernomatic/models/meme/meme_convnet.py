"""
MEME_CONVNET
The convolutional network from the imgflip meme generator
"""

import torch
import torch.nn as nn
from lernomatic.models.common import LernomaticModel


class MemeConvnet(LernomaticModel):
    def __init__(self, num_embeddings:int, embed_dim:int, **kwargs) -> None:
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.num_bn_features:int = kwargs.pop('num_bn_features', 100)
        self.dropout_rate:float = kwargs.pop('dropout_rate', 0.25)

        # setup network
        self.embedding = nn.Embedding(self.num_embeddings, self.embed_dim)
        # NOTE: for now I will just take the model definition directly from the
        # imgflip model
        self.layer1 = nn.Sequential(
            nn.Conv1d(1024, 5, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.num_bn_features),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(1024, 5, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.num_bn_features),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(1024, 5, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.num_bn_features),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(1024, 5, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.num_bn_features),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_rate)
        )
        self.final = nn.Linear(self.num_labels)

    def __repr__(self) -> str:
        return 'MemeConvnet'

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        y = self.layer1(X)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.final(y)

        return y
