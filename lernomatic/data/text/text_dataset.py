"""
TEXT_DATASET
Datasets for text data

Stefan Wong 2019
"""

import h5py
import json
import torch
from torch.utils import data

from lernomatic.data.text import corpus
from lernomatic.data.text import word_map


