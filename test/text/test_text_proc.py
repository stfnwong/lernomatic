"""
TEST_TEXT_PROC
Unit tests for text processing.

Stefan Wong 2019
"""

import sys
import argparse
import unittest
# units under test
from lernomatic.data.text import word_map
from lernomatic.data.text import corpus
from lernomatic.data.text import text_proc

# debug
from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()

class TestTextProc(unittest.TestCase):
    def setUp(self):
        self.verbose   = GLOBAL_OPTS['verbose']
        self.test_file = '/mnt/ml-data/datasets/shakespear_corpus.txt'

    def test_word_level_proc(self):
        print('======== TestTextProc.test_word_level_proc ')

        test_output_file = 'data/word_level_proc_test.h5'
        wmap = word_map.WordMap()
        with open(self.test_file, 'r') as fp:
            for line in fp:
                words = line.split()
                wmap.update(words)

        wmap.generate()
        # get processor
        word_level_proc = text_proc.TextWordLevelProc(
            wmap,
            verbose=self.verbose
        )
        word_level_proc.proc(self.test_file, test_output_file)

        print('======== TestTextProc.test_word_level_proc <END>')


    def test_char_level_proc(self):
        print('======== TestTextProc.test_char_level_proc ')

        test_output_file = 'data/char_level_proc_test.h5'
        wmap = word_map.WordMap()
        with open(self.test_file, 'r') as fp:
            for line in fp:
                words = line.split()
                wmap.update(words)

        wmap.generate()
        # get processor
        word_level_proc = text_proc.TextCharLevelProc(
            verbose=self.verbose
        )
        word_level_proc.proc(self.test_file, test_output_file)


        print('======== TestTextProc.test_char_level_proc <END>')



# TODO : check args

# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--draw-plot',
                        action='store_true',
                        default=False,
                        help='Draw plots'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of worker processes to use for HDF5 load'
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--device-id',
                        type=int,
                        default=0,
                        help='Device to use for tests (default : -1)'
                        )
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    arg_vals = vars(args)
    for k, v in arg_vals.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print('-------- GLOBAL OPTS (%s) --------' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()
