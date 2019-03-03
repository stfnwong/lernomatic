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

def main():

    #fig, ax = plt.subplots()
    fig, ax = vis_loss_history.get_figure_subplots(2)
    history = torch.load(GLOBAL_OPTS['input'])

    loss_history = history['loss_history'][0 : history['loss_iter']]
    if GLOBAL_OPTS['verbose']:
        print('%d training iterations ' % history['loss_iter'])
        print('Current epoch : %d' % history['cur_epoch'])
        print('%d iters per epoch' % history['iter_per_epoch'])

    if 'acc_history' in history:
        acc_history = history['acc_history'][0 : history['acc_iter']]
        if GLOBAL_OPTS['verbose']:
            print('%d accuracy iterations ' % history['acc_iter'])
            print('Max accuracy : %.3f ' % max(history['acc_history'][0: history['acc_iter']]))
    else:
        acc_history = None
        if GLOBAL_OPTS['verbose']:
            print('No acc_history in file [%s]' % GLOBAL_OPTS['input'])
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
        iter_per_epoch = history['iter_per_epoch'],
        cur_epoch = history['cur_epoch'],
        acc_unit = 'BLEU-4'
    )

    fig.tight_layout()
    if GLOBAL_OPTS['outfile'] is not None:
        fig.savefig(GLOBAL_OPTS['outfile'], bbox_inches='tight')
    else:
        plt.show()



def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('input',
                        type=str,
                        default=None,
                        help='Checkpoint file to read'
                        )
    parser.add_argument('--outfile',
                        type=str,
                        default=None,
                        help='File to write plot output to (default: None)'
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

        print(' ')

    main()
