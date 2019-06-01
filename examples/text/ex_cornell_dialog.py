"""
EX_CORNELL_DIALOG
Examples that use the Cornell Dialog Corpus

Stefan Wong 2019
"""


import argparse

from lernomatic.data.text import cornell_movie
from lernomatic.data.text import vocab


GLOBAL_OPTS = dict()



def print_lines(filename:str, n:int=10) -> None:
    with open(filename, 'rb') as fp:
        lines = fp.readlines()
        for line in lines[:n]:
            print(line)


def main() ->None:


    corpus_lines_filename = '/mnt/ml-data/datasets/cornell_movie_dialogs_corpus/movie_lines.txt'
    corpus_conversations_filename = '/mnt/ml-data/datasets/cornell_movie_dialogs_corpus/movie_conversations.txt'
    corpus_csv_outfile = 'data/cornell_corpus_out.csv'
    print_lines(corpus_lines_filename, 10)

    mcorpus = cornell_movie.CornellMovieCorpus(verbose=True)
    mcorpus.load_lines(corpus_lines_filename)
    mcorpus.load_conversations(corpus_conversations_filename)
    mcorpus.extract_sent_pairs()
    mcorpus.write_csv(corpus_csv_outfile)




    print('OK')


def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
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
