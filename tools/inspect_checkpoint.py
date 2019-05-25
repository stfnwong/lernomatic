"""
INSPECT_CHECKPOINT

Stefan Wong 2019
"""

import argparse
import sys
import torch


GLOBAL_OPTS = dict()
TOOL_MODES = ()


def dump_items(data:dict, level:int = 0, max_level:int=5,) -> None:
    if level >= max_level:
        return
    if level > 0:
        tab_chars = "\t" * level
    else:
        tab_chars = ""
    for k, v in data.items():
        print('%s[%s] : %s' % (tab_chars, str(k), type(v)))
        if type(v) is dict:
            dump_items(v, level+1, max_level)


def main() -> None:
    if GLOBAL_OPTS['device_id'] < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % int(GLOBAL_OPTS['device_id']))

    checkpoint_data = torch.load(GLOBAL_OPTS['input'])
    dump_items(
        checkpoint_data,
        level=0,
        max_level=GLOBAL_OPTS['max_level'])


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        help='Input file'
                        )
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Device id to map checkpoint tensors to. -1 indicates CPU (default: -1)'
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
