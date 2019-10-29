"""
TEST_QRSEQ2SEQ_TRAINER
Unit tests for QRSeq2SeqTrainer module

Stefan Wong 2019
"""
import sys
import argparse
import unittest
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# units under test
from lernomatic.train.text import qr_seq2seq_trainer
from lernomatic.models.text import seq2seq
from lernomatic.data.text import qr_dataset
from lernomatic.data.text import vocab

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


# TODO: more extensive typing?
def get_models(hidden_size:int,
               enc_num_layers:int,
               dec_num_layers:int,
               num_words:int,
               embedding:nn.Module=None) -> tuple:
    encoder = seq2seq.EncoderRNN(
        hidden_size,
        num_layers = enc_num_layers,
        num_words  = num_words,
        dropout = 0.0,
        embedding = embedding
    )
    decoder = seq2seq.LuongAttenDecoderRNN(
        hidden_size,
        num_words,
        num_layers = dec_num_layers,
        dropout = 0.0,
        embedding = embedding
    )

    return (encoder, decoder)


def get_embed_layer(num_words:int, hidden_size:int, filename:str=None) -> nn.Module:
    embedding = nn.Embedding(num_words, hidden_size)
    if filename is not None:
        embedding.load_state_dict(filename)

    return embedding


class TestQRSeq2SeqTrainer(unittest.TestCase):
    def setUp(self):
        self.hidden_size = 512
        self.enc_num_layers = 1
        self.dec_num_layers = 2
        # path to data
        self.train_data_path = 'hdf5/cornell-movie-train.h5'
        self.voc_data_path = 'hdf5/test_vocab.json'

    def test_save_load(self):
        print('======== TestQRSeq2SeqTrainer.test_save_load ')

        test_checkpoint_file = 'checkpoint/qr_seq2seq_trainer_test_checkpoint.pkl'
        test_history_file = 'checkpoint/qr_seq2seq_trainer_test_history.pkl'
        test_batch_size = 16
        test_num_epochs = 4

        # get some data
        train_dataset = qr_dataset.QRDataset(
            self.train_data_path
        )

        if GLOBAL_OPTS['common_embed']:
            embedding_layer = get_embed_layer(
                train_dataset.get_num_words(),
                self.hidden_size
            )
        else:
            embedding_layer = None

        # get models
        encoder, decoder = get_models(
            self.hidden_size,
            self.enc_num_layers,
            self.dec_num_layers,
            train_dataset.get_num_words(),
            embedding = embedding_layer
        )

        # get vocab
        voc = vocab.Vocabulary('Seq2Seq Vocab')
        voc.load(self.voc_data_path)

        # get a trainer
        src_trainer = qr_seq2seq_trainer.QRSeq2SeqTrainer(
            voc,        # where to get voc from?
            encoder,
            decoder,
            train_dataset = train_dataset,
            # training options
            device_id     = GLOBAL_OPTS['device_id'],
            batch_size    = test_batch_size,
            num_epochs    = test_num_epochs,
            print_every   = GLOBAL_OPTS['print_every'],
            save_every    = 0
        )

        src_trainer.train()
        print('Saving checkpoint data to file [%s] ' % str(test_checkpoint_file))
        src_trainer.save_checkpoint(test_checkpoint_file)
        src_trainer.save_history(test_history_file)

        # Get a new trainer object and load params there
        dst_trainer = qr_seq2seq_trainer.QRSeq2SeqTrainer(
            None,
            None
        )
        print('Loading checkpoint data from file [%s] ' % str(test_checkpoint_file))
        dst_trainer.load_checkpoint(test_checkpoint_file)

        # Check model parameters
        self.assertIsNotNone(dst_trainer.encoder)
        self.assertIsNotNone(dst_trainer.decoder)

        print('======== TestQRSeq2SeqTrainer.test_save_load <END>')



# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--common-embed',
                        action='store_true',
                        default=False,
                        help='Use the same embedding layer for encoder and decoder'
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
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Device to use for tests (default : -1)'
                        )
    # display options
    parser.add_argument('--print-every',
                        type=int,
                        default=20,
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
