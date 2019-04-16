"""
WORDMAP_PROBE
Examine a wordmap object

Stefan Wong 2019
"""

import sys
import argparse
from lernomatic.data.text import word_map

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def main() -> None:
    wmap = word_map.WordMap()
    wmap.load(GLOBAL_OPTS['input'])

    print('Loaded WordMap [%s] containing %d words' %\
          (str(GLOBAL_OPTS['input']), len(wmap))
    )

    if GLOBAL_OPTS['t'] is not None:
        tok = wmap.tok2word(GLOBAL_OPTS['t'])
        if tok == wmap.get_unk():
            tok = '<unk>'
        print('Token [%s] : %s' % (str(GLOBAL_OPTS['t']), tok))

    if GLOBAL_OPTS['w'] is not None:
        word = wmap.word2tok(GLOBAL_OPTS['w'])
        if word == wmap.get_unk():
            word = '<unk>'
        print('Word  [%s] : %s' % (str(GLOBAL_OPTS['w']), word))


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        help='Input file'
                        )
    parser.add_argument('--mode',
                        choices=['inspect',  'load', 'find'],
                        default='inspect',
                        help='Select the tool mode from one of inspect, load, find'
                        )
    parser.add_argument('-t',
                        type=int,
                        default=None,
                        help='Lookup token in wordmap'
                        )
    parser.add_argument('-w',
                        type=str,
                        default=None,
                        help='Lookup word in wordmap'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
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

    if GLOBAL_OPTS['verbose']:
        print('==== GLOBAL OPTS (%s) ====' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('\t [%s] : %s' % (str(k), str(v)))

    main()
