"""
CHEKPOINT
Plot data from a checkpoint

Stefan Wong 2018
"""

import argparse
import torch
import matplotlib.pyplot as plt
from lernomatic.vis import vis_loss_history
from lernomatic.vis.gan import vis_gan_loss

# debug
#

GLOBAL_OPTS = dict()
VALID_TOOL_MODES = ('show', 'probe', 'gan')


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
    if 'test_loss_history' in history:
        test_loss_history = history['test_loss_history'][0 : history['test_loss_iter']]
        if GLOBAL_OPTS['verbose']:
            print('%d test loss iterations' % history['test_loss_iter'])
    else:
        test_loss_history = None
        if GLOBAL_OPTS['verbose']:
            print('No test_loss_history in file [%s]' % GLOBAL_OPTS['input'])

    if GLOBAL_OPTS['print_loss']:
        print(str(loss_history))

    if GLOBAL_OPTS['print_acc']:
        print(str(acc_history))

    # plot the visualization
    vis_loss_history.plot_train_history_2subplots(
        ax,
        loss_history,
        test_loss_history = test_loss_history,
        acc_history = acc_history,
        loss_title = GLOBAL_OPTS['title'],

    if GLOBAL_OPTS['test_loss_history_key'] in history:
        test_loss_history = history[GLOBAL_OPTS['test_loss_history_key']][0 : history['test_loss_iter']]
    else:
        test_loss_history = None

    if acc_history is not None:
        fig, ax = vis_loss_history.get_figure_subplots(2)
    else:
        fig, ax = vis_loss_history.get_figure_subplots(1)

    vis_loss_history.plot_train_history_2subplots(
        ax,
        loss_history,
        test_loss_curve = test_loss_history,
        acc_curve = acc_history,
        title = GLOBAL_OPTS['title'],
        iter_per_epoch = history['iter_per_epoch'],
        cur_epoch = history['cur_epoch'],
        acc_unit = 'BLEU-4'
    )

    fig.tight_layout()
    if GLOBAL_OPTS['outfile'] is not None:
        fig.savefig(GLOBAL_OPTS['outfile'], bbox_inches='tight')
    else:
        plt.show()


    if GLOBAL_OPTS['plot_filename'] is not None:
        fig.tight_layout()
        fig.savefig(GLOBAL_OPTS['plot_filename'])
    else:
        plt.show()


# The idea of this mode is just to print the contents (keys, really) in
# the history file
def probe() -> None:
    history = torch.load(GLOBAL_OPTS['input'])
    print('Contents of file [%s]' % str(GLOBAL_OPTS['input']))
    for k, v in history.items():
        print('  [%s] : %s' % (str(k), type(v)))


def gan() -> None:
    fig, ax = plt.subplots()
    history = torch.load(GLOBAL_OPTS['input'])

    # Do a quick check that the keys we need are actually in the history file
    if GLOBAL_OPTS['g_loss_history_key'] not in history:
        raise ValueError('No key [%s] in history file [%s]' %\
                    (str(GLOBAL_OPTS['g_loss_history_key']), str(GLOBAL_OPTS['input']))
        )

    if GLOBAL_OPTS['d_loss_history_key'] not in history:
        raise ValueError('No key [%s] in history file [%s]' %\
                    (str(GLOBAL_OPTS['d_loss_history_key']), str(GLOBAL_OPTS['input']))
        )

    g_loss_history = history[GLOBAL_OPTS['g_loss_history_key']][0 : history['loss_iter']]
    d_loss_history = history[GLOBAL_OPTS['d_loss_history_key']][0 : history['loss_iter']]

    vis_gan_loss.plot_gan_loss(
        ax,
        g_loss_history,
        d_loss_history,
        title = GLOBAL_OPTS['title'],
        iter_per_epoch = history['iter_per_epoch'],
        cur_epoch = history['cur_epoch']
    )

    if GLOBAL_OPTS['print_loss']:
        print('\t Generator loss [%s] :' % str(GLOBAL_OPTS['g_loss_history_key']))
        print(str(g_loss_history))
        print('\t Discriminator loss [%s] :' % str(GLOBAL_OPTS['d_loss_history_key']))
        print(str(d_loss_history))

    if GLOBAL_OPTS['plot_filename'] is not None:
        fig.tight_layout()
        fig.savefig(GLOBAL_OPTS['plot_filename'])
    else:
        plt.show()


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
                        choices = VALID_TOOL_MODES,
                        default='show',
                        help='Tool mode. (default: show)'
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
    parser.add_argument('--plot-filename',
                        type=str,
                        default='loss_history.png',
                        help='Plot output file name'
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
    # GAN options
    parser.add_argument('--g-loss-history-key',
                        type=str,
                        default='g_loss_history',
                        help='Key that identifies generator loss history (default: g_loss_history)'
                        )
    parser.add_argument('--d-loss-history-key',
                        type=str,
                        default='d_loss_history',
                        help='Key that identifies discriminator loss history (default: d_loss_history)'
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
            print('\t[%s] : %s' % (str(k), str(v)))
    print('')

    if GLOBAL_OPTS['mode'] == 'show':
        show()
    elif GLOBAL_OPTS['mode'] == 'probe':
        probe()
    elif GLOBAL_OPTS['mode'] == 'gan':
        gan()
    else:
        raise ValueError('Invalid tool mode [%s]' % str(GLOBAL_OPTS['mode']))
