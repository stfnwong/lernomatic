"""
TEST_ENC_DEC_TRAINER
Unit tests for the new Encoder/Decoder trainer

Stefan Wong 2019
"""
import sys
import argparse
import unittest
import torch
import matplotlib.pyplot as plt
# units under test

from lernomatic.train.text import enc_dec_trainer
from lernomatic.models.text import enc_dec_atten
# other data stuff
from lernomatic.data.text import vocab

# debug
from pudb import set_trace; set_trace()





GLOBAL_OPTS = dict()

# TODO : model helper function?


class TestEncDecTrainer(unittest.TestCase):
    def setUp(self):
        # TODO : need to generate a vocab herre
        self.embed_size      = 256
        self.enc_num_layers  = 1
        self.dec_num_layers  = 1
        self.hidden_size     = 512
        self.test_batch_size = 32

        self.test_num_words  = 11        # for synthetic copy task

    # Checkpoint, history test
    #def test_save_load(self):
    #    print('======== TestEncDecTrainer.test_save_load ')

    #    print('======== TestEncDecTrainer.test_save_load <END>')


    def test_copy_task(self):
        print('======== TestEncDecTrainer.test_copy_task ')

        # get some models
        encoder = enc_dec_atten.Encoder(
            self.test_num_words,
            self.hidden_size,
            num_layers = self.enc_num_layers
        )

        decoder = enc_dec_atten.Decoder(
            self.embed_size,
            self.hidden_size,
            num_layers = self.dec_num_layers
        )

        generator = enc_dec_atten.Generator(
            self.hidden_size,
            self.test_num_words
        )

        # get a trainer
        trainer = enc_dec_trainer.EncDecTrainer(
            encoder,
            decoder,
            generator,
            num_words = self.test_num_words,       # TODO: replace with vocab
            batch_size = self.test_batch_size,
            embed_dim  = self.embed_size,
            device_id = GLOBAL_OPTS['device_id'],
        )
        trainer.train()


        print('======== TestEncDecTrainer.test_copy_task <END>')



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
            print('\t[%s] : %s' % (str(k), str(v)))

    sys.argv[1:] = args.unittest_args
    unittest.main()
