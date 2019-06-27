"""
INSPECT_LR_FIND
Dump the contents of an LRFinder's state to console

Stefan Wong 2019
"""

import sys
import argparse
import matplotlib.pyplot as plt

from lernomatic.param import lr_common
# vis stuff
from lernomatic.vis import vis_loss_history


GLOBAL_OPTS = dict()
VALID_TOOL_MODES = ('dump', 'plot')


def dump() -> None:
    """
    DUMP
    Dump parameters to console
    """
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

    print('%s parameter history' % repr(lr_finder))
    for k in skip_keys:
        if k not in params:
            continue
        print('\t[%s] contains %d elements' % (str(k), len(params[k])))


def plot() -> None:
    """
    PLOT
    Draw standard plots of find() results
    """
    lr_finder = lr_common.lr_finder_auto_load(GLOBAL_OPTS['input'])
    if GLOBAL_OPTS['lr_acc_title'] is not None:
        lr_acc_title = GLOBAL_OPTS['lr_acc_title']
    else:
        lr_acc_title  = '[' + str(lr_finder.lr_select_method) + '] learning rate vs acc (log)'

    if GLOBAL_OPTS['lr_loss_title'] is not None:
        lr_loss_title = GLOBAL_OPTS['lr_loss_title']
    else:
        lr_loss_title = '[' + str(lr_finder.lr_select_method) + '] learning rate vs loss (log)'

    lr_fig, lr_ax = vis_loss_history.get_figure_subplots(2)
    lr_finder.plot_lr_vs_acc(lr_ax[0], lr_acc_title, log=True)
    lr_finder.plot_lr_vs_loss(lr_ax[1], lr_loss_title, log=True)
    # save
    lr_fig.tight_layout()

    if GLOBAL_OPTS['draw']:
        plt.show()
    else:
        lr_fig.savefig(GLOBAL_OPTS['plotfile'], bbox_inches='tight')


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
    parser.add_argument('--draw',
                        action='store_true',
                        default=False,
                        help='Draw plot directly using matplotlib (default: False)'
                        )
    parser.add_argument('--mode',
                        type=str,
                        default='dump',
                        help='Tool mode. Must be one of %s (default: dump)' % str(VALID_TOOL_MODES)
                        )
    # plot options
    parser.add_argument('--lr-acc-title',
                        type=str,
                        default=None,
                        help='LR acc plot title (default: None)'
                        )
    parser.add_argument('--lr-loss-title',
                        type=str,
                        default=None,
                        help='LR Loss plot title (default: None)'
                        )
    parser.add_argument('--plotfile',
                        type=str,
                        default='lr_finder.png',
                        help='Path to file to save plot to (default: lr_finder.png)'
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
    elif GLOBAL_OPTS['mode'] == 'plot':
        plot()
    else:
        raise ValueError('Unsupported tool mode [%s]' % str(GLOBAL_OPTS['mode']))
