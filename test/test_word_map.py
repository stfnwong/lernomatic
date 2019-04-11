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

# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()


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


class TestWordMap(unittest.TestCase):

    def test_generate_word_map(self):
        print('======== TestWordMap.test_generate_word_map ')

        # get all the datasets
        coco_data_splits = []
        for split in ('train', 'test', 'val'):
            coco_data_splits.append(
                get_data_split(
                    GLOBAL_OPTS['coco_json'],
                    GLOBAL_OPTS['data_root'],
                    split,
                    max_items = 0,
                    max_capt_len = 64,
                    verbose = GLOBAL_OPTS['verbose']
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
            self.assertIsInstance(captions, list)
            self.assertIsInstance(captions[0], list)
            for n, cap in enumerate(captions):
                print('Updating word map with caption [%d/%d]' %\
                      (n+1, len(captions)), end='      \r'
                )
                wmap.update(cap)

        print('\n OK')

        wmap.generate()
        self.assertEqual(0, wmap.get_pad())

        test_string = 'the quick brown fox jumps over the lazy dog'.split(' ')
        print('Converting string to tokens....')
        tokens = []
        for word in test_string:
            tok = wmap.word2tok(word)
            self.assertNotEqual(wmap.get_unk(), tok)
            tokens.append(tok)
            print('%s ' % tok)

        print('Converting tokens to string...')
        for tok in tokens:
            word = wmap.tok2word(tok)
            print('%s ' % word)


        print('======== TestWordMap.test_generate_word_map <END>')

    def test_save_load(self):
        print('======== TestWordMap.test_save_load ')

        # get all the datasets
        coco_data_splits = []
        for split in ('train', 'test', 'val'):
            coco_data_splits.append(
                get_data_split(
                    GLOBAL_OPTS['coco_json'],
                    GLOBAL_OPTS['data_root'],
                    split,
                    max_items = 0,
                    max_capt_len = 64,
                    verbose = GLOBAL_OPTS['verbose']
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
            self.assertIsInstance(captions, list)
            self.assertIsInstance(captions[0], list)
            for n, cap in enumerate(captions):
                print('Updating word map with caption [%d/%d]' %\
                      (n+1, len(captions)), end='      \r'
                )
                wmap.update(cap)

        wmap.generate()
        self.assertEqual(len(wmap.word_map), len(wmap.map_word))
        test_wmap_file = 'data/test_wordmap.json'
        print('Saving word map to file [%s]' % str(test_wmap_file))
        wmap.save(test_wmap_file)

        load_wmap = word_map.WordMap()
        load_wmap.load(test_wmap_file)
        print('Checking word map loaded from file [%s]' % str(test_wmap_file))
        self.assertEqual(len(wmap), len(load_wmap))

        print('Checking word map...')
        for n, (k, v) in enumerate(wmap.word_map.items()):
            print('Checking key %s [%d / %d]' % (str(k), n+1, len(wmap)), end='        \r')
            self.assertEqual(v, load_wmap.word_map[k])

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
            self.assertEqual(v, load_wmap.map_word[k])

        print('\n DONE')

        print('Removing test file [%s]' % str(test_wmap_file))
        os.remove(test_wmap_file)

        print('======== TestWordMap.test_save_load <END>')

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    # coco path info
    parser.add_argument('--coco-json',
                        type=str,
                        default='/mnt/ml-data/datasets/COCO/dataset_coco.json',
                        help='Path to COCO json file'
                        )
    parser.add_argument('--data-root',
                        type=str,
                        default='/mnt/ml-data/datasets/COCO/',
                        help='Path to root of COCO dataset'
                        )
    parser.add_argument('--train-data-path',
                        type=str,
                        default='/mnt/ml-data/datasets/COCO/coco-train.h5',
                        help='Path to COCO train dataset'
                        )
    parser.add_argument('--test-data-path',
                        type=str,
                        default='/mnt/ml-data/datasets/COCO/coco-test.h5',
                        help='Path to COCO train dataset'
                        )
    parser.add_argument('--val-data-path',
                        type=str,
                        default='/mnt/ml-data/datasets/COCO/coco-val.h5',
                        help='Path to COCO train dataset'
                        )
    # set up args for unittest
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    arg_vals = vars(args)
    for k, v in arg_vals.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print('-------- GLOBAL OPTS (%s) --------' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()
