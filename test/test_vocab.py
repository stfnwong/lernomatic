"""
TEST_VOCAB
Unit tests for vocabulary object

Stefan Wong 2019
"""

import sys
import argparse
import unittest
# units under test
from lernomatic.data.text import cornell_movie
from lernomatic.data.text import vocab

# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.num_sample_qr = 20
        self.test_vocab_file = 'data/test_vocab.json'
        self.corpus_lines_filename = GLOBAL_OPTS['data_root'] +\
            'cornell_movie_dialogs_corpus/movie_lines.txt'
        self.corpus_conversations_filename = GLOBAL_OPTS['data_root'] \
            + 'cornell_movie_dialogs_corpus/movie_conversations.txt'

    def test_all(self):
        print('======== TestVocabulary.test_all ')

        mcorpus = cornell_movie.CornellMovieCorpus(
            self.corpus_lines_filename,
            self.corpus_conversations_filename,
            verbose=True
        )
        qr_pairs = mcorpus.extract_sent_pairs(max_length=GLOBAL_OPTS['max_qr_len'])

        src_vocab = vocab.Vocabulary('Source Vocabulary')
        for n, pair in enumerate(qr_pairs):
            print('Adding pair [%d/%d] to vocab' % (n+1, len(qr_pairs)), end='\r')
            src_vocab.add_sentence(pair.query)
            src_vocab.add_sentence(pair.response)

        # Lookup the builtin tokens
        self.assertEqual(0, src_vocab.lookup_word('<pad>'))


        # save the vocab to file
        src_vocab.save(self.test_vocab_file)
        # load from file into new vocab object
        dst_vocab = vocab.Vocabulary('Destination Vocabulary')
        dst_vocab.load(self.test_vocab_file)

        self.assertEqual(len(src_vocab), len(dst_vocab))
        # check each word, mapping in turn


        print('======== TestVocabulary.test_all <END>')


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
