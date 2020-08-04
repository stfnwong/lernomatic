"""
INSPECT_CHECKPOINT

Stefan Wong 2019
"""

import argparse
import sys
import torch

GLOBAL_OPTS = dict()


# Recursively dump data from checkpoint
def dump_items(data:dict,
               level:int = 0,
               max_level:int=5,
               show_data:bool=False) -> None:
    if level >= max_level:
        return
    if level > 0:
        tab_chars = "\t" * level
    else:
        tab_chars = ""
    for k, v in data.items():
        if show_data is True and (isinstance(v, str) or isinstance(v, int) or isinstance(v, float)):
            print('%s[%s] : %s (%s)' % (tab_chars, str(k), type(v), str(v)))
        else:
            print('%s[%s] : %s' % (tab_chars, str(k), type(v)))
        if type(v) is dict:
            dump_items(v, level+1, max_level, show_data)


# Program entry
def main() -> None:
    checkpoint_data = torch.load(GLOBAL_OPTS['input'])
    dump_items(
        checkpoint_data,
        level     = 0,
        max_level = GLOBAL_OPTS['max_level'],
        show_data  = GLOBAL_OPTS['show_data']
    )


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        help='Input file'
                        )
    parser.add_argument('--max-level',
                        type=int,
                        default=3,
                        help='Max recursion depth to print checkpoint data (default: 3)'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--show-data',
                        action='store_true',
                        default=False,
                        help='Print the value of checkpoint items with simple datatypes (int, str, float, etc)'
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
