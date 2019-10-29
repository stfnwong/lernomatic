"""
TEST_QR_DATASET
Unit tests for QRDataset object

Stefan Wong 2019
"""
import sys
import argparse
import unittest
import torch
import h5py
import numpy as np
from tqdm import tqdm
# modules under test
from lernomatic.data.text import qr_batch
from lernomatic.data.text import cornell_movie
from lernomatic.data.text import vocab
from lernomatic.data.text import qr_pair
from lernomatic.data.text import qr_dataset

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


# util function to createa
# TODO : re-use the class in qr_proc.py
def create_qr_dataset(qr_pairs:list,
                      voc:vocab.Vocabulary,
                      filename:str,
                      vec_len:int=32,
                      delimiter:str=' ') -> None:
    with h5py.File(filename, 'w') as fp:
        queries = fp.create_dataset(
            'query',
            (len(qr_pairs), vec_len),
            dtype=np.int32
        )
        responses = fp.create_dataset(
            'response',
            (len(qr_pairs), vec_len),
            dtype=np.int32
        )
        qlen = fp.create_dataset(
            'qlength',
            (len(qr_pairs),1),
            dtype=np.int32
        )
        rlen = fp.create_dataset(
            'rlength',
            (len(qr_pairs),1),
            dtype=np.int32
        )
        queries.attrs['num_words'] = len(voc)
        queries.attrs['vec_len'] = vec_len

        for elem_idx, pair in enumerate(tqdm(qr_pairs, unit='Q/R pairs')):
            # convert query
            enc_query = [voc.lookup_word(w) for w in pair.query_to_list()] +\
                        [voc.get_eos()]
            qlen[elem_idx] = pair.query_len()
            if len(enc_query) < vec_len:
                enc_query.extend([voc.get_pad()] * (vec_len-len(enc_query)))
            elif len(enc_query) > vec_len:
                qlen[elem_idx] = vec_len
                enc_query = enc_query[0 : vec_len-1]
                enc_query.extend([voc.get_eos()])
            queries[elem_idx] = enc_query

            # convert response
            enc_response = [voc.lookup_word(w) for w in pair.response_to_list()] +\
                        [voc.get_eos()]
            rlen[elem_idx] = pair.response_len()
            if len(enc_response) < vec_len:
                enc_response.extend([voc.get_pad()] * (vec_len-len(enc_response)))
            elif len(enc_response) > vec_len:
                rlen[elem_idx] = vec_len
                enc_response = enc_response[0 : vec_len-1]
                enc_response.extend([voc.get_eos()])
            responses[elem_idx] = enc_response


class TestQRDataset(unittest.TestCase):
    def setUp(self):
        self.vocab_name = 'QRDataset Test Vocab'
        self.test_batch_size = 8
        self.test_max_length = 10
        self.test_max_pairs = 64
        self.qr_dataset_path = 'hdf5/qr_dataset.h5'

        # cornell corpus paths
        self.corpus_lines_filename = GLOBAL_OPTS['data_root'] +\
        'cornell_movie_dialogs_corpus/movie_lines.txt'
        self.corpus_conversations_filename = GLOBAL_OPTS['data_root'] +\
        'cornell_movie_dialogs_corpus/movie_conversations.txt'

    def test_batch(self):
        print('======== TestQRDataset.test_batch ')

        mcorpus = cornell_movie.CornellMovieCorpus(
            self.corpus_lines_filename,
            self.corpus_conversations_filename,
            verbose=True
        )
        qr_pairs = mcorpus.extract_sent_pairs(max_length=self.test_max_length)

        # get a new vocab object
        mvocab = vocab.Vocabulary(self.vocab_name)
        for n, pair in enumerate(qr_pairs):
            print('Adding pair [%d / %d] to vocab' % (n+1, len(qr_pairs)), end='\r')
            mvocab.add_sentence(pair.query)
            mvocab.add_sentence(pair.response)


        print('Creating new QRDataset [%s]' % str(self.qr_dataset_path))
        create_qr_dataset(
            qr_pairs[0:self.test_max_pairs],
            mvocab,
            self.qr_dataset_path,
            vec_len = self.test_max_length
        )

        print('Loading dataset [%s] from disk' % str(self.qr_dataset_path))
        train_dataset = qr_dataset.QRDataset(
            self.qr_dataset_path
        )
        self.assertEqual(len(mvocab), train_dataset.get_num_words())
        # Test this inside a loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = self.test_batch_size,
            shuffle = False
        )

        for batch_idx, (query, qlen, response, rlen) in enumerate(train_loader):
            if batch_idx > 4:
                break
            print('\nbatch %d' % batch_idx)

            # sort the data in descending order for pack_padded_sequence()
            qlen, qlen_sort_idx = qlen.squeeze(1).sort(dim=0, descending=True)
            rlen, rlen_sort_idx = rlen.squeeze(1).sort(dim=0, descending=True)
            query = query[qlen_sort_idx]
            response = response[rlen_sort_idx]

            query = query.transpose(0, 1)
            response = response.transpose(0, 1)

            print(query)
            print(response)

            print('Query vectors as strings...')
            for q in range(query.shape[0]):
                print(q, vocab.vec2sentence(query[q], mvocab))

            print('Response vectors as strings...')
            for r in range(response.shape[0]):
                print(r, vocab.vec2sentence(response[r], mvocab))

            ref_query, ref_lengths, ref_response, mask, max_target_len = qr_batch.batch_convert(
                mvocab,
                qr_pairs[batch_idx : (batch_idx+1) * self.test_batch_size],
            )

            print('Reference query vectors as strings...')
            for q in range(ref_query.shape[0]):
                print(q, vocab.vec2sentence(ref_query[q], mvocab))

            print('Reference response vectors as strings...')
            for q in range(ref_response.shape[0]):
                print(q, vocab.vec2sentence(ref_response[q], mvocab))

            # assert on data
            #self.assertEqual(ref_query.shape, query.shape)
            #self.assertEqual(ref_response.shape, response.shape)

            #for row in range(ref_query.shape[0]):
            #    for col in range(ref_query.shape[1]):
            #        print('Checking element <%d,%d> (ref_query [%d], query [%d]' %\
            #              (row, col, ref_query[row][col], query[row][col]), end='\r'
            #        )
            #        self.assertTrue(torch.eq(ref_query[row][col], query[row][col]))

            #for row in range(ref_response.shape[0]):
            #    for col in range(ref_response.shape[1]):
            #        print('Checking element <%d,%d> (ref_response [%d], response [%d]' %\
            #              (row, col, ref_response[row][col], response[row][col]), end='\r'
            #        )
            #        self.assertTrue(torch.eq(ref_response[row][col], response[row][col]))

        print('======== TestQRDataset.test_batch <END>')



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
