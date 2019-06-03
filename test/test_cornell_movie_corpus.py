"""
TEST_CORNELL_MOVIE_CORPUS
Unit tests for handling the Cornell Movie Corpus Data

Stefan Wong 2019
"""


import sys
import argparse
import unittest
import numpy as np
# modules under test
from lernomatic.data.text import batch
from lernomatic.data.text import cornell_movie
from lernomatic.data.text import vocab


GLOBAL_OPTS = dict()


def print_lines(filename:str, n:int=10) -> None:
    with open(filename, 'rb') as fp:
        lines = fp.readlines()
        for line in lines[:n]:
            print(line)


class TestCornellMovieCorpus(unittest.TestCase):

    def setUp(self):
        self.corpus_lines_filename         = GLOBAL_OPTS['data_root'] + 'cornell_movie_dialogs_corpus/movie_lines.txt'
        self.corpus_conversations_filename = GLOBAL_OPTS['data_root'] + 'cornell_movie_dialogs_corpus/movie_conversations.txt'
        self.qr_pair_csv_file              = 'data/cornell_corpus_out.csv'
        self.test_max_length = 20
        self.num_sample_lines = 20
        #self.verbose = GLOBAL_OPTS['verbose']

    def test_create_corpus(self):
        print('\n======== TestCornellMovieCorpus.test_create_corpus ')

        mcorpus = cornell_movie.CornellMovieCorpus(
            self.corpus_lines_filename,
            self.corpus_conversations_filename,
            verbose=True
        )
        print('Created new corpus object')
        print(mcorpus)
        self.assertGreater(mcorpus.get_num_lines(), 0)
        self.assertGreater(mcorpus.get_num_conversations(), 0)
        self.assertEqual(304713, mcorpus.get_num_lines())
        self.assertEqual(83097, mcorpus.get_num_conversations())

        print('======== TestCornellMovieCorpus.test_create_corpus <END>')

    def test_qa_pair_gen(self):
        print('\n======== TestCornellMovieCorpus.test_qr_pair_gen')

        # create a new corpus
        mcorpus = cornell_movie.CornellMovieCorpus(
            self.corpus_lines_filename,
            self.corpus_conversations_filename,
            verbose=True
        )

        print('Extracting sentence pairs....')
        qr_pairs = mcorpus.extract_sent_pairs(max_length=0)
        self.assertEqual(221282, len(qr_pairs))

        # Write pairs to csv and check that reading them back they are equal
        cornell_movie.qr_pairs_to_csv(
            self.qr_pair_csv_file,
            qr_pairs,
            verbose = True
        )

        csv_qr_pairs = cornell_movie.qr_pair_proc_from_csv(
            self.qr_pair_csv_file,
            max_length =  0,
            verbose = True
        )

        print('%d Q/R pairs generated from corpus' % len(qr_pairs))
        print('%d Q/R pairs read from file [%s]' % (len(csv_qr_pairs), str(self.qr_pair_csv_file)))
        self.assertEqual(len(qr_pairs), len(csv_qr_pairs))

        for n, (p1, p2) in enumerate(zip(qr_pairs, csv_qr_pairs)):
            print('Checking Q/R pair [%d / %d]' % (n+1, len(qr_pairs)), end='\r')
            self.assertEqual(p1, p2)

        print('\n OK')

        print('======== TestCornellMovieCorpus.test_qr_pair_gen <END>')



class TestCornellMovieVocab(unittest.TestCase):
    def setUp(self):
        self.corpus_lines_filename         = GLOBAL_OPTS['data_root'] + 'cornell_movie_dialogs_corpus/movie_lines.txt'
        self.corpus_conversations_filename = GLOBAL_OPTS['data_root'] + 'cornell_movie_dialogs_corpus/movie_conversations.txt'
        self.test_max_length = 10
        self.test_batch_size = 16

    def test_gen_cornell_vocab(self):
        print('\n======== TestCornellMovieVocab.test_gen_cornell_vocab ')

        mcorpus = cornell_movie.CornellMovieCorpus(
            self.corpus_lines_filename,
            self.corpus_conversations_filename,
            verbose=True
        )
        qr_pairs = mcorpus.extract_sent_pairs(max_length=self.test_max_length)

        # get a new vocab object
        mvocab = vocab.Vocabulary('Cornell Movie Vocab')
        for n, pair in enumerate(qr_pairs):
            print('Adding pair [%d / %d] to vocab' % (n+1, len(qr_pairs)), end='\r')
            mvocab.add_sentence(pair.query)
            mvocab.add_sentence(pair.response)

        print('\n OK')
        print(mvocab)
        # if test_max_length = 20, then we expect there to be 33021 words in
        # vocab
        self.assertEqual(17993, len(mvocab))

        print('======== TestCornellMovieVocab.test_gen_cornell_vocab <END>')


    def test_vocab_batch(self):
        print('\n======== TestCornellMovieVocab.test_vocab_batch ')

        mcorpus = cornell_movie.CornellMovieCorpus(
            self.corpus_lines_filename,
            self.corpus_conversations_filename,
            verbose=True
        )
        qr_pairs = mcorpus.extract_sent_pairs(max_length=self.test_max_length)

        # get a new vocab object
        mvocab = vocab.Vocabulary('Cornell Movie Vocab')
        for n, pair in enumerate(qr_pairs):
            print('Adding pair [%d / %d] to vocab' % (n+1, len(qr_pairs)), end='\r')
            mvocab.add_sentence(pair.query)
            mvocab.add_sentence(pair.response)

        inp_batch_data, inp_lengths, out_batch_data, mask, max_target_len = batch.batch_convert(
            mvocab,
            qr_pairs[0 : self.test_batch_size],
        )

        print('in batch shape :', inp_batch_data.shape)
        print('out batch shape :', out_batch_data.shape)
        print('input lengths :', inp_lengths)
        print('mask :', mask)
        print('max_target_len :', max_target_len)

        # Note that for larger values of self.test_max_length  its not
        # guaranteed that the tensors will have dimensions as large as
        # self.test_max_length, owing to the fact that many of the sentences
        # in the corpus may not be long enough
        self.assertEqual(self.test_batch_size, inp_batch_data.shape[1])
        self.assertEqual(self.test_max_length, out_batch_data.shape[0])
        self.assertEqual(self.test_batch_size, out_batch_data.shape[1])
        self.assertEqual(self.test_max_length, max_target_len)
        self.assertEqual(self.test_batch_size, inp_lengths.shape[0])

        print('======== TestCornellMovieVocab.test_vocab_batch <END>')



# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--data-root',
                        type=str,
                        default='/mnt/ml-data/datasets/',
                        help='Path to root of dataset'
                        )

    parser.add_argument('--min-word-freq',
                        type=int,
                        default=5,
                        help='Minimum number of times a word can occur before it is pruned from the vocabulary (default: 5)'
                        )
    parser.add_argument('--max-qr-len',
                        type=int,
                        default=20,
                        help='Maximum length in words that a query or response may be (default: 10)'
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
