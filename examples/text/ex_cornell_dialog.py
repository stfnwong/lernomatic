"""
EX_CORNELL_DIALOG
Examples that use the Cornell Dialog Corpus

Stefan Wong 2019
"""


import argparse

from lernomatic.data.text import cornell_movie
from lernomatic.data.text import vocab
from lernomatic.data.text import batch


GLOBAL_OPTS = dict()



def print_lines(filename:str, n:int=10) -> None:
    with open(filename, 'rb') as fp:
        lines = fp.readlines()
        for line in lines[:n]:
            print(line)


def main() ->None:


    corpus_lines_filename = GLOBAL_OPTS['data_root'] + 'cornell_movie_dialogs_corpus/movie_lines.txt'
    corpus_conversations_filename = GLOBAL_OPTS['data_root'] + 'cornell_movie_dialogs_corpus/movie_conversations.txt'
    qr_pairs_csv_file = 'data/cornell_corpus_out.csv'
    print('Some sample lines from corpus...')
    print_lines(corpus_lines_filename, 10)

    mcorpus = cornell_movie.CornellMovieCorpus(
        corpus_lines_filename,
        corpus_conversations_filename,
        verbose=True
    )
    qr_pairs = mcorpus.extract_sent_pairs()

    # TODO: why not do this in memory? Note that that answer might be that in
    # some cases the corpus is too large to fit in memory
    #mcorpus.write_csv(qr_pairs_csv_file)
    cornell_movie.qr_pairs_to_csv(
        qr_pair_csv_file,
        qr_pairs,
        verbose = True
    )


    # get a list of query/response pairs
    print('Generating Query/Response pairs from file [%s]' % str(qr_pairs_csv_file))
    qr_pairs = cornell_movie.qr_pair_proc_from_csv(
        qr_pairs_csv_file,
        max_length = GLOBAL_OPTS['max_qr_len'],
        verbose = True
    )

    mvocab = vocab.Vocabulary('test_vocab')
    for n, pair in enumerate(qr_pairs):
        print('Adding pair [%d/%d] to vocab' % (n+1, len(qr_pairs)), end='\r')
        mvocab.add_sentence(pair.query)
        mvocab.add_sentence(pair.response)
    print('\n Created new vocabulary of %d words' % len(mvocab))
    print('Pruning words that appear fewer than %d times' % GLOBAL_OPTS['min_word_freq'])
    mvocab.trim_freq(GLOBAL_OPTS['min_word_freq'])
    print('Created new vocabulary:')
    print(mvocab)




    print('OK')


def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
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

    return parser


if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS ---- ')
        for k,v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    main()
