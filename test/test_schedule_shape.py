"""
TEST_SCHEDULE_SHAPE
Draw plots of scheduler shapes

Stefan Wong 2019
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from lernomatic.train import schedule

from typing import Dict


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



def test_schedules() -> None:
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
    lr_out:Dict[str, float] = dict()

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

        """
        To have useful asserts here we need either
        1) Schedule output data that we do a compare on
        2) A parameterized equation of the schedule data that we can fit against
        in order to say something like 'the output curve is parametrically
        equivalent to the expected curve'.
        """
