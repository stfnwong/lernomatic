"""
TEST_VOCAB
Unit tests for vocabulary object

Stefan Wong 2019
"""

import sys
import argparse
import unittest
# units under test
from lernomatic.data.text import vocab

# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()


class TestVocabulary(unittest.TestCase):
    def test_gen_vocab(self):
        print('======== TestVocabulary.test_gen_voacb ')

        v = vocab.Vocabulary('Test Vocabulary')

        print('======== TestVocabulary.test_gen_voacb <END>')


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
