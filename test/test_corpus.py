"""
TEST_CORPUS

Stefan Wong 2019
"""

import sys
import argparse
import unittest

# modules under test
from lernomatic.data.text import corpus
from lernomatic.data.text import word_map

# debug
#from pudb import set_trace; set_trace()


class TestCorpus(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']

    def test_init_corpus(self):
        print('======== TestCorpus.test_init_corpus ')

        test_text_file = '/mnt/ml-data/datasets/grimm/grimm.txt'

        wmap = word_map.WordMap()
        with open(test_text_file, 'r') as fp:
            for line in fp:
                words = line.split()
                wmap.update(words)
        wmap.generate()
        text_corpus = corpus.Corpus(wmap)

        token_tensor = text_corpus.tokenize_file(test_text_file, update_map=True)
        print(token_tensor)
        print(len(token_tensor))

        print('======== TestCorpus.test_init_corpus <END>')


GLOBAL_OPTS = dict()


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
