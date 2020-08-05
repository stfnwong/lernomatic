"""
TEST_WORD_MAP
Unit test for WordMap object

Stefan Wong 2019
"""

import os
import sys
import unittest
import argparse
import numpy as np
import matplotlib.pyplot as plt

# modules under test
from lernomatic.data.text import word_map
from lernomatic.data.coco import coco_data


def get_data_split(json_path:str,
                data_root:str,
                split_name:str,
                max_items:int=0,
                max_capt_len:int=64,
                verbose:bool=False) -> coco_data.COCODataSplit:

    s = coco_data.COCODataSplit(
        json_path,
        data_root,
        split_name   = split_name,
        max_items    = max_items,
        max_capt_len = max_capt_len,
        verbose      = verbose
    )

    return s


class TestWordMap:
    verbose          = True
    test_max_batches = 128
    draw_plot        = False
    coco_json        = '/mnt/ml-data/datasets/COCO/dataset_coco.json'
    data_root        = '/mnt/ml-data/datasets/COCO/'

    def test_generate_word_map(self) -> None:
        # get all the datasets
        coco_data_splits = []
        for split in ('train', 'test', 'val'):
            coco_data_splits.append(
                get_data_split(
                    self.coco_json,
                    self.data_root,
                    split,
                    max_items = 0,
                    max_capt_len = 64,
                    verbose = self.verbose
                )
            )

        print('Creating word map objects...')
        wmap = word_map.WordMap()
        for split in coco_data_splits:
            split.create_split()

        # We expect a list of lists to come out of split.get_captions(). Each
        # caption is a list, and there is a list of all the captions for that
        # split.
        for s in coco_data_splits:
            captions = s.get_captions()
            assert isinstance(captions, list)
            assert isinstance(captions[0], list)
            for n, cap in enumerate(captions):
                print('Updating word map with caption [%d/%d]' %\
                      (n+1, len(captions)), end='      \r'
                )
                wmap.update(cap)

        print('\n OK')

        wmap.generate()
        assert wmap.get_pad() == 0

        test_string = 'the quick brown fox jumps over the lazy dog'.split(' ')
        print('Converting string to tokens....')
        tokens = []
        for word in test_string:
            tok = wmap.word2tok(word)
            assert wmap.get_unk() != tok
            tokens.append(tok)
            print('%s ' % tok)

        print('Converting tokens to string...')
        for tok in tokens:
            word = wmap.tok2word(tok)
            print('%s ' % word)


    def test_save_load(self) -> None:
        # get all the datasets
        coco_data_splits = []
        for split in ('train', 'test', 'val'):
            coco_data_splits.append(
                get_data_split(
                    self.coco_json,
                    self.data_root,
                    split,
                    max_items = 0,
                    max_capt_len = 64,
                    verbose = self.verbose
                )
            )

        print('Creating word map objects...')
        wmap = word_map.WordMap()
        for split in coco_data_splits:
            split.create_split()

        # We expect a list of lists to come out of split.get_captions(). Each
        # caption is a list, and there is a list of all the captions for that
        # split.
        for s in coco_data_splits:
            captions = s.get_captions()
            assert isinstance(captions, list)
            assert isinstance(captions[0], list)
            for n, cap in enumerate(captions):
                print('Updating word map with caption [%d/%d]' %\
                      (n+1, len(captions)), end='      \r'
                )
                wmap.update(cap)

        wmap.generate()
        assert len(wmap.word_map) ==  len(wmap.map_word)
        test_wmap_file = 'data/test_wordmap.json'
        print('Saving word map to file [%s]' % str(test_wmap_file))
        wmap.save(test_wmap_file)

        load_wmap = word_map.WordMap()
        load_wmap.load(test_wmap_file)
        print('Checking word map loaded from file [%s]' % str(test_wmap_file))
        assert len(wmap) == len(load_wmap)

        print('Checking word map...')
        for n, (k, v) in enumerate(wmap.word_map.items()):
            print('Checking key %s [%d / %d]' % (str(k), n+1, len(wmap)), end='        \r')
            assert v == load_wmap.word_map[k]

        print('\n DONE')
        print('Checking map word...')

        print('wmap keys')
        item_limit = 16
        for n, (k, v) in enumerate(wmap.map_word.items()):
            print(str(k), ':', str(v), '(', type(k), ':', type(v), ')')
            if n > item_limit:
                break

        print('load_wmap keys')
        for n, (k, v) in enumerate(load_wmap.map_word.items()):
            print(str(k), ':', str(v), '(', type(k), ':', type(v), ')')
            if n > item_limit:
                break

        print(load_wmap.map_word[1])

        for n, (k, v) in enumerate(wmap.map_word.items()):
            print('Checking key %s [%d / %d]' % (str(k), n+1, len(wmap)), end='        \r')
            assert v == load_wmap.map_word[k]

        print('\n DONE')

        print('Removing test file [%s]' % str(test_wmap_file))
        os.remove(test_wmap_file)
