"""
INSPECT_LR_FIND
Dump the contents of an LRFinder's state to console

Stefan Wong 2019
"""

import sys
import argparse

from lernomatic.param import lr_common

GLOBAL_OPTS = dict()
VALID_TOOL_MODES = ('dump')


def dump() -> None:

    # by default we don't dump any of the history since that would create a
    # large amount of scrollback
    skip_keys = ('smooth_loss_history', 'acc_history', 'log_lr_history', 'loss_grad_history')
    lr_finder = lr_common.lr_finder_auto_load(GLOBAL_OPTS['input'])
    params = lr_finder.get_params()

    print('\n%s paramters from file [%s]' % (repr(lr_finder), str(GLOBAL_OPTS['input'])))
    for k, v in params.items():
        if k in skip_keys:
            continue
        print('\t [%s] : %s' % (str(k), str(v)))



def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        help='Input file'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--mode',
                        type=str,
                        default='dump',
                        help='Tool mode. Must be one of %s (default: dump)' % str(VALID_TOOL_MODES)
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
        print('==== GLOBAL OPTS ==== (%s)' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    if GLOBAL_OPTS['mode'] == 'dump':
        dump()
    else:
        raise ValueError('Unsupported tool mode [%s]' % str(GLOBAL_OPTS['mode']))
