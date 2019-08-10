"""
GAN_LOSS_HISTORY
Plot loss history of a GAN model

Stefan Wong 2019
"""

import argparse
import torch
import matplotlib.pyplot as plt
from lernomatic.vis.gan import vis_gan_loss

GLOBAL_OPTS = dict()
VALID_TOOL_MODES = ('show', 'probe')



def show() -> None:
    fig, ax = plt.subplots()
    history = torch.load(GLOBAL_OPTS['input'])




def probe() -> None:
    pass


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('checkpoint',
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
    # names of the models in checkpoint
    parser.add_argument('--generator-key',
                        type=str,
                        default='generator',
                        help='Name of generator model (default: generator)'
                        )
    parser.add_argument('--discriminator-key',
                        type=str,
                        default='discriminator',
                        help='Name of discriminator model (default: discriminator)'
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



if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

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
