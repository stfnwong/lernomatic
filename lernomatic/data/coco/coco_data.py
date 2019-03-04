"""
COCO_DATA
Utils for dealing with the COCO dataset

Stefan Wong 2018
"""

import os
import json
import h5py
from tqdm import tqdm
import numpy as np
from random import seed, choice, sample
from imageio import imread

from lernomatic.data import data_split
from lernomatic.data.text import word_map

# debug
#from pudb import set_trace; set_trace()

# TODO : get rid of this...?
def gen_coco_data_split(coco_json, data_root, split_name='train', verbose=True):

    split_data = data.DataSplit(split_name=split_name)
    wm = word_map.WordMap()

    with open(coco_json, 'r') as fp:
        coco_data_json = json.load(fp)

    for n, img in enumerate(coco_json_data):
        captions = []
        for c in img['sentences']:
            pass

    return split_data


class CaptionDataSplit(object):
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


class COCODataSplit(object):
    """
    COCODATASPLIT
    Encapsulates a single data split from the COCO dataset
    """
    def __init__(self, coco_json, data_root, **kwargs):
        self.coco_json = coco_json
        self.data_root = data_root

        self.max_items    = kwargs.pop('max_items', 0)
        self.max_capt_len = kwargs.pop('max_capt_len', 32)
        self.capt_per_img = kwargs.pop('capt_per_img', 5)
        self.img_dim      = kwargs.pop('img_dim', 256)
        self.seed         = kwargs.pop('seed', None)
        self.verbose      = kwargs.pop('verbose', False)
        split_name        = kwargs.pop('split_name', 'train')

        # Do a type check
        if type(self.coco_json) is not str:
            raise TypeError('coco_json must be a path to json file of type str')
        if type(self.data_root) is not str:
            raise TypeError('data_root must be a path to json file of type str')

        #self.split_data = data_split.DataSplit(split_name=split_name)
        self.split_data = CaptionDataSplit(split_name = split_name)

        # Try to load the data from json
        try:
            with open(self.coco_json, 'r') as fp:
                self.data = json.load(fp)
        except Exception as e:
            print(e)
            raise ValueError('Failed to init %s' % repr(self))

        # iteration index
        self.idx = 0

    def __repr__(self):
        return 'COCODataSplit (%d items)' % len(self.split_data)

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.split_data)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.split_data):
            impath = self.split_data.image_paths[self.idx]
            caption = self.split_data.captions[self.idx]
            self.idx += 1
            return (impath, caption)
        else:
            raise StopIteration

    def get_param_dict(self):
        param = dict()
        param['split_data']   = self.split_data._get_param_dict()
        param['max_items']    = self.max_items
        param['max_capt_len'] = self.max_capt_len
        param['capt_per_img'] = self.capt_per_img
        param['img_dim']      = self.img_dim
        param['verbose']      = self.verbose

        return param

    def set_param_dict(self, param):
        self.split_data.set_param_from_dict(param['split_data'])
        self.max_item     = param['max_items']
        self.max_capt     = param['max_capt_len']
        self.capt_per_img = param['capt_per_img']
        self.img_dim      = param['img_dim']
        self.verbose      = param['verbose']

    def save(self, fname):
        params = self.get_param_dict()
        with open(fname, 'w') as fp:
            json.dump(params, fp)

    def load(self, fname):
        with open(fname, 'r') as fp:
            params = json.load(fp)
        self.set_param_dict(params)

    def get_num_paths(self):
        return self.split_data.get_num_paths()

    def get_num_captions(self):
        return self.split_data.get_num_captions()

    def get_split_name(self):
        return self.split_data.split_name

    def get_captions(self):
        return self.split_data.get_captions()

    def get_image_paths(self):
        return self.split_data.get_image_captions()

    def create_split(self):
        """
        CREATE_SPLIT
        Read through the COCO data and collect all data in this split
        """
        for n, img in enumerate(self.data['images']):
            capt = list()
            path = os.path.join(self.data_root, img['filepath'], img['filename'])
            # Grab raw captions and save
            for c in img['sentences']:
                if len(c['tokens']) <= self.max_capt_len:
                   capt.append(c['tokens'])
            if img['split'] in self.split_data.split_name:
                self.split_data.add_path(path)
                self.split_data.add_caption(capt)
                self.split_data.add_id(img['imgid'])

            if self.verbose:
                print('\t Checking image [%d / %d] (%d captions found, %d images in split <%s>)' % \
                      (n, len(self.data['images']), len(c['tokens']), len(self.split_data), self.split_data.split_name), \
                      end='\r')

            # Break early if we have enough items
            if self.max_items > 0:
                if len(self.split_data) >= self.max_items:
                    break

        if self.verbose:
            print(' ')
