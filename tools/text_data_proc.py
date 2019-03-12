"""
TEXT_DATA_PROC
Process a text dataset

Stefan Wong 2019
"""

import argparse
from lernomatic.data.text import word_map


GLOBAL_OPTS = dict()


def main():

    wmap = word_map.WordMap()
    # read in some files



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root',
                        type=str,
                        default='/mnt/ml-data/datasets/joke-dataset',
                        help='Path to root of text dataset'
                        )

    return parser


if __name__ == '__main__':
    parser = arg_parser()
    args = vars(parser.parse_args())

    for k, v in args.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['input'] is None:
        print('ERROR: no input file specified.')
        sys.exit(1)

    main()
