"""
TEST_CORNELL_MOVIE_CORPUS
Unit tests for handling the Cornell Movie Corpus Data

Stefan Wong 2019
"""


import sys
import argparse
import unittest
# modules under test
from lernomatic.data.text import batch
from lernomatic.data.text import cornell_movie


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
        print('======== TestCornellMovieCorpus.test_create_corpus ')

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
        print('======== TestCornellMovieCorpus.test_qr_pair_gen')

        # create a new corpus
        mcorpus = cornell_movie.CornellMovieCorpus(
            self.corpus_lines_filename,
            self.corpus_conversations_filename,
            verbose=True
        )
        mcorpus.extract_sent_pairs()

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

        # print the first 10 of each set of pairs
        print('Q/R pairs from Corpus')
        for n in range(10):
            print(qr_pairs[n])

        print('Q/R pairs from *.csv')
        for n in range(10):
            print(csv_qr_pairs[n])

        for n, (p1, p2) in enumerate(zip(qr_pairs, csv_qr_pairs)):
            print('Checking Q/R pair [%d / %d]' % (n+1, len(qr_pairs)), end='\r')
            self.assertEqual(p1, p2)

        print('\n OK')


        print('======== TestCornellMovieCorpus.test_qr_pair_gen <END>')




#class TestBatch(unittest.TestCase):
#
#    def setUp(self):
#        self.batch_size = 5
#
#    def test_create_batch(self):
#        print('======== TestBatch.test_create_batch <END>')



GLOBAL_OPTS = dict()

# modules under test
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
