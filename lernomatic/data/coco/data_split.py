"""
DATA_SPLIT
Data split object for COCO Image captioning

Stefan Wong 2018
"""

# TODO: if this is going to live on as a specialized class then it should
# probably get a new name, and perhaps be re-tooled to sub-class DataSplit

#import os
import json
#import h5py
import numpy as np
#from collections import Counter
#from random import seed, choice, sample
#from scipy.misc import imread
#from scipy.misc import imresize
#from tqdm import tqdm
# torch stuff
import torch

# debug
#from pudb import set_trace; set_trace()

class DataSplit(object):
    """
    SPLITDATA
    Holds information about a data split
    """
    def __init__(self, split_name='unknown'):
        self.image_paths = list()
        self.captions    = list ()
        self.elem_ids    = list()
        self.split_name  = split_name
        # iteration index
        self.idx         = 0

    def __len__(self):
        return len(self.image_paths)

    def __str__(self):
        s = []
        s.append('DataSplit <%s> (%d items)\n' % (self.split_name, len(self)))
        return ''.join(s)

    def __getitem__(self, idx):
        return (self.image_paths[idx], self.captions[idx])

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.image_paths):
            path    = self.image_paths[self.idx]
            caption = self.captions[self.idx]
            elem_id = self.elem_ids[self.idx]
            self.idx += 1
            return (path, caption, elem_id)
        else:
            raise StopIteration

    def _get_param_dict(self):
        param = dict()
        param['image_paths'] = self.image_paths
        param['captions']    = self.captions
        param['elem_ids']    = self.elem_ids
        param['split_name']  = self.split_name
        return param

    def _set_param_from_dict(self, param):
        self.image_paths = param['image_paths']
        self.captions    = param['captions']
        self.elem_ids    = param['elem_ids']
        self.split_name  = param['split_name']

    def reset(self):
        self.image_paths = list()
        self.captions    = list()
        self.elem_ids    = list()

    def get_num_paths(self):
        return len(self.image_paths)

    def get_num_captions(self):
        return len(self.captions)

    def add_caption(self, c):
        self.captions.append(c)

    def add_path(self, p):
        self.image_paths.append(p)

    def add_id(self, i):
        self.elem_ids.append(i)

    def get_captions(self):
        return self.captions

    def get_image_paths(self):
        return self.image_paths

    def get_elem_ids(self):
        return self.elem_ids

    def save(self, fname):
        param = self._get_param_dict()
        with open(fname, 'w') as fp:
            json.dump(param, fp)

    def load(self, fname):
        with open(fname, 'r') as fp:
            param = json.load(fp)
        self._set_param_from_dict(param)


def init_embeddings(emb):
    """
    Fills embedding tensor with values from the uniform distribution.
    """
    bias = np.sqrt(3.0 / emb.size(1))
    torch.nn.init.uniform_(emb, -bias, bias)

def load_embeddings(emb_file, word_map):
    """
    Create an embedding tensor for the specified word map for loading
    into the model

    Args:
        emb_file - file containing embeddings in GloVe format.
        word_map - dict containing word map

    Return:
        embeddedings in order of word map, dimensions of embeddings
    """

    # find embed dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1
    vocab = set(word_map.keys())

    # create tensor to hold embeddings and init
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embeddings()

    # read the embedding file
    print('Reading embeddings from file %s...' % str(emb_file))
    for line in open(emb_file, 'r'):
        line = line.split(' ')
        emb_word = line[0]
        # TODO : unpack this lambda
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))
        # ignore word if not in train vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim
