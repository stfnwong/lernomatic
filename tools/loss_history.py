"""
CHEKPOINT
Plot data from a checkpoint

Stefan Wong 2018
"""

import argparse
import torch
import matplotlib.pyplot as plt
from lernomatic.vis import vis_loss_history

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()
VALID_TOOL_MODES = ('show', 'probe')


def show() -> None:
    fig, ax = plt.subplots()
    history = torch.load(GLOBAL_OPTS['input'])

    if GLOBAL_OPTS['loss_history_key'] not in history:
        raise ValueError('No key [%s] in history file [%s] (try using --mode=probe)' %\
                         (str(GLOBAL_OPTS['loss_history_key']), str(GLOBAL_OPTS['input']))
        )

    loss_history = history[GLOBAL_OPTS['loss_history_key']][0 : history['loss_iter']]

    if GLOBAL_OPTS['acc_history_key'] in history:
        acc_history = history[GLOBAL_OPTS['acc_history_key']][0 : history['loss_iter']]
    else:
        acc_history = None

    if GLOBAL_OPTS['test_loss_history_key'] in history:
        test_loss_history = history[GLOBAL_OPTS['test_loss_history_key']][0 : history['test_loss_iter']]
    else:
        test_loss_history = None

    #if GLOBAL_OPTS['verbose']:
    #    print('Checkpoint [%s] current epoch : %d' % (str(GLOBAL_OPTS['input']), t.cur_epoch))
    if GLOBAL_OPTS['verbose']:
        print('%d training iterations in loss history file [%s]' %\
              (int(history['loss_iter']), str(GLOBAL_OPTS['input'])))

    vis_loss_history.plot_train_history_2subplots(
        ax,
        loss_history,
        test_loss_curve = test_loss_history,
        acc_curve = acc_history,
        title = GLOBAL_OPTS['title'],
        iter_per_epoch = history['iter_per_epoch'],
        cur_epoch = history['cur_epoch']
    )

    if GLOBAL_OPTS['print_loss']:
        print(str(loss_history))

    if GLOBAL_OPTS['print_acc']:
        print(str(acc_history))

    plt.show()


# The idea of this mode is just to print the contents (keys, really) in
# the history file
def probe() -> None:
    history = torch.load(GLOBAL_OPTS['input'])
    for k, v in history.items():
        print('[%s] : %s' % (str(k), type(v)))


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('input',
                        type=str,
                        default=None,
                        help='Checkpoint file to read'
                        )
    parser.add_argument('--mode',
                        type=str,
                        default='show',
                        help='Tool mode. Must be one of %s (default: show)' % str(VALID_TOOL_MODES)
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--title',
                        type=str,
                        default=None,
                        help='Title for plot output'
                        )
    # names of loss keys
    parser.add_argument('--loss-history-key',
                        type=str,
                        default='loss_history',
                        help='Key that identifies loss history (default: loss_history)'
                        )
    parser.add_argument('--test-loss-history-key',
                        type=str,
                        default='test_loss_history',
                        help='Key that identifies test loss history (default: test_loss_history)'
                        )
    parser.add_argument('--acc-history-key',
                        type=str,
                        default='acc_history',
                        help='Key that identifies loss history (default: acc_history)'
                        )
    # other opts
    parser.add_argument('--print-loss',
                        default=False,
                        action='store_true',
                        help='Print loss history to console'
                        )
    parser.add_argument('--print-acc',
                        default=False,
                        action='store_true',
                        help='Print acc history to console'
                        )

    return parser


# Entry point
if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    # TODO : probably should make this positional later
    if GLOBAL_OPTS['input'] is None:
        raise ValueError('No input file specified')

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS ---- ')
        for k,v in GLOBAL_OPTS.items():
            print('%s : %s' % (str(k), str(v)))

    if GLOBAL_OPTS['mode'] == 'show':
        show()
    elif GLOBAL_OPTS['mode'] == 'probe':
        probe()
    else:
        raise ValueError('Invalid tool mode [%s]' % str(GLOBAL_OPTS['mode']))
