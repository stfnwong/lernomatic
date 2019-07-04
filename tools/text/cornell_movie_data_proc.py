"""
CORNELL_MOVIE_DATA_PROC
Pre-processing for Cornell Movie Corpus

Stefan Wong 2019
"""

import sys
import argparse
from lernomatic.data.text import qr_batch
from lernomatic.data.text import qr_dataset
from lernomatic.data.text import qr_pair
from lernomatic.data.text import qr_split
from lernomatic.data.text import qr_proc
from lernomatic.data.text import cornell_movie
from lernomatic.data.text import vocab


GLOBAL_OPTS = dict()


def main() -> None:
    corpus_lines_filename         = GLOBAL_OPTS['data_root'] + GLOBAL_OPTS['corpus_lines_filename']
    corpus_conversations_filename = GLOBAL_OPTS['data_root'] + GLOBAL_OPTS['corpus_conv_filename']

    mcorpus = cornell_movie.CornellMovieCorpus(
        corpus_lines_filename,
        corpus_conversations_filename,
        verbose=GLOBAL_OPTS['verbose']
    )
    qr_pairs = mcorpus.extract_sent_pairs(max_length=GLOBAL_OPTS['max_qr_len'])

    # get a new vocab object
    mvocab = vocab.Vocabulary('Cornell Movie Vocab')
    for n, pair in enumerate(qr_pairs):
        print('Adding pair [%d / %d] to vocab' % (n+1, len(qr_pairs)), end='\r')
        mvocab.add_sentence(pair.query)
        mvocab.add_sentence(pair.response)

    if GLOBAL_OPTS['min_word_freq'] > 0:
        mvocab.trim_freq(GLOBAL_OPTS['min_word_freq'])

    if GLOBAL_OPTS['vocab_outfile'] is not None:
        if GLOBAL_OPTS['verbose']:
            print('Saving vocabulary to file [%s]' % str(GLOBAL_OPTS['vocab_outfile']))
        mvocab.save(GLOBAL_OPTS['vocab_outfile'])

    # generate data splits
    splitter = qr_split.QRDataSplitter(
        split_names  = GLOBAL_OPTS['split_names'],
        split_ratios = GLOBAL_OPTS['split_ratios'],
        verbose      = GLOBAL_OPTS['verbose']
    )
    splits = splitter.gen_splits(qr_pairs)

    if GLOBAL_OPTS['verbose']:
        for n, s in enumerate(splits):
            print('Split %d <%s> contains %d items' % (n, s.split_name, len(s)))

    processor = qr_proc.QRDataProc(
        vec_len = GLOBAL_OPTS['max_qr_len'],
        verbose = GLOBAL_OPTS['verbose']
    )

    for n, split in enumerate(splits):
        print('Processing split <%s> [%d / %d]' % (split.split_name, n+1, len(splits)))
        split_filename = 'hdf5/cornell-movie-%s.h5' % (str(split.split_name))
        processor.proc(mvocab, split, split_filename)

    print('\n DONE')


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    # output files
    parser.add_argument('--vocab-outfile',
                        type=str,
                        default=None,
                        help='If specified, write the generated vocab to this file (default: None)'
                        )
    # data options
    parser.add_argument('--data-root',
                        type=str,
                        default='/mnt/ml-data/datasets/',
                        help='Path to root of dataset'
                        )
    parser.add_argument('--corpus-lines-filename',
                        type=str,
                        default='cornell_movie_dialogs_corpus/movie_lines.txt',
                        help='Path to movie lines file'
                        )
    parser.add_argument('--corpus-conv-filename',
                        type=str,
                        default='cornell_movie_dialogs_corpus/movie_conversations.txt',
                        help='Path to movie conversations file'
                        )
    # vocab options
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
    # split options
    parser.add_argument('--split-names',
                        type=str,
                        default='train,test,val',
                        help='Comma seperated list of names for each split (default: train,test,val)'
                        )
    parser.add_argument('--split-ratios',
                        type=str,
                        default='0.7,0.15,0.15',
                        help='Comma seperated list of ratios for each split. Must sum to 1 (default: 0.7,0.15,0.15)'
                        )

    return parser


if __name__ == '__main__':

    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print('-------- GLOBAL OPTS (%s) --------' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    GLOBAL_OPTS['split_names'] = GLOBAL_OPTS['split_names'].split(',')
    split_ratios = GLOBAL_OPTS['split_ratios'].split(',')
    split_ratio_floats = []

    for s in split_ratios:
        split_ratio_floats.append(float(s))

    GLOBAL_OPTS['split_ratios'] = split_ratio_floats

    if len(GLOBAL_OPTS['split_names']) != len(GLOBAL_OPTS['split_ratios']):
        raise ValueError('Number of split rations must equal number of split names')

    if sum(split_ratio_floats) > 1.0:
        raise ValueError('Sum of split ratios cannot exceed 1.0')

    main()
