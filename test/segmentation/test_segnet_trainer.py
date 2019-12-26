"""
TEST_SEGNET_TRAINER
Test SegnetTrainer object.

Stefan Wong 2019
"""

import sys
import argparse
import unittest
import torch
import matplotlib.pyplot as plt
# units under test
from lernomatic.train.segmentation import segnet_trainer
from lernomatic.models.segmentation import segnet
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


class TestSegnetTrainer(unittest.TestCase):
    def setUp(self):
        self.verbose = GLOBAL_OPTS['verbose']
        self.resnet_depth = 28
        self.test_batch_size = 64
        self.test_num_epochs = 2
        self.test_learning_rate = 0.001

    def test_save_load_checkpoint(self):
        print('======== TestSegnetTrainer.test_save_load_checkpoint ')
        # TODO : what training data do I use?

        encoder = segnet.SegNetEncoder()
        decoder = segnet.SegNetDecoder()


        print('======== TestSegnetTrainer.test_save_load_checkpoint <END>')



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
                        default=8,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Device to use for tests (default : -1)'
                        )
    # display options
    parser.add_argument('--print-every',
                        type=int,
                        default=100,
                        help='Print output every time this number of iterations has elapsed'
                        )
    # checkpoint options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='checkpoint/',
                        help='Directory to save checkpoints to'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='resnet-trainer-test',
                        help='String to prefix to checkpoint files'
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
