"""
TEST_SCHEDULE_SHAPE
Draw plots of scheduler shapes

Stefan Wong 2019
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from lernomatic.train import schedule


GLOBAL_OPTS = dict()

def get_scheduler(lr_min:float,
                  lr_max:float,
                  stepsize:int = 1000,
                  sched_type='TriangularScheduler') -> schedule.LRScheduler:
    if sched_type is None:
        return None

    if not hasattr(schedule, sched_type):
        raise ValueError('Unknown scheduler type [%s]' % str(sched_type))

    lr_sched_obj = getattr(schedule, sched_type)
    lr_scheduler = lr_sched_obj(
        lr_min = lr_min,
        lr_max = lr_max,
        stepsize = stepsize
    )

    return lr_scheduler


def plot_schedule(lr_out:np.ndarray,
                  title:str,
                  fname:str) -> None:

    fig, ax = plt.subplots()
    ax.plot(lr_out)
    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Learning Rate')
    fig.savefig(fname)

def main() -> None:
    schedulers = [
        'TriangularScheduler',
        'InvTriangularScheduler',
        'Triangular2Scheduler',
        'ExponentialDecayScheduler',
        'WarmRestartScheduler',
        'TriangularExpScheduler',
        'Triangular2ExpScheduler',
        #'DecayWhenAcc',
        #'TriangularDecayWhenAcc',
    ]
    lr_out = dict()

    for sched in schedulers:
        sched_object = get_scheduler(
            GLOBAL_OPTS['lr_min'],
            GLOBAL_OPTS['lr_max'],
            GLOBAL_OPTS['sched_stepsize'],
            sched
        )

        print('Running scheduler [%s] for %d iterations' % (str(sched), GLOBAL_OPTS['num_iters']))
        lr_out[sched] = np.zeros(GLOBAL_OPTS['num_iters'])
        for n in range(GLOBAL_OPTS['num_iters']):
            lr_out[sched][n] = sched_object.get_lr(n)

        # Make a plot and save it
        plot_schedule(lr_out[sched],
                      str(sched),
                      str(GLOBAL_OPTS['figure_dir']) + str(sched) + '-schedule.png'
        )


def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--draw-plot',
                        default=False,
                        action='store_true',
                        help='Display plots'
                        )
    parser.add_argument('--figure-dir',
                        type=str,
                        default='./',
                        help='Path to figure directory (default: ./)'
                        )
    parser.add_argument('--num-iters',
                        type=int,
                        default=8000,
                        help='Number of iterations to test for'
                        )
    # Learning rate finder options
    parser.add_argument('--find-print-every',
                        type=int,
                        default=20,
                        help='How often to print output from learning rate finder'
                        )
    parser.add_argument('--find-num-epochs',
                        type=int,
                        default=8,
                        help='Maximum number of epochs to attempt to find learning rate'
                        )
    parser.add_argument('--find-explode-thresh',
                        type=float,
                        default=4.5,
                        help='Threshold at which to stop increasing learning rate'
                        )
    parser.add_argument('--lr-min',
                        type=float,
                        default=2e-4,
                        help='Minimum range to search for learning rate'
                        )
    parser.add_argument('--lr-max',
                        type=float,
                        default=1e-1,
                        help='Maximum range to search for learning rate'
                        )
    parser.add_argument('--lr-select-method',
                        type=str,
                        default='min_loss',
                        help='Method to use for selecting LR range'
                        )
    # Schedule options
    parser.add_argument('--exp-decay',
                        type=float,
                        default=0.001,
                        help='Exponential decay term'
                        )
    parser.add_argument('--sched-stepsize',
                        type=int,
                        default=4000,
                        help='Size of step for learning rate scheduler'
                        )
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS ---- ')
        for k,v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    main()
