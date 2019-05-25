"""
TEST_CAPTION_INFER
Unit tests for Caption Inferrer

Stefan Wong 2019
"""

import sys
import argparse
import unittest
import torch
# unit(s) under test
from lernomatic.models import common
from lernomatic.models import cifar
from lernomatic.train import cifar_trainer
from lernomatic.infer import infer_caption

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


class TestCaptionInferrer(unittest.TestCase):

    def test_infer_init(self):
        print('======== TestCaptionInferrer.test_infer_init ')





        print('======== TestCaptionInferrer.test_infer_init <END>')


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
                        help='Number of worker processes to use for reading HDF5'
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


