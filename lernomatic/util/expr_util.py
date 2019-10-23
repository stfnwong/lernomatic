"""
EXPR_UTIL
Experiment utils

Stefan Wong 2019
"""

from lernomatic.train import trainer
from lernomatic.train import schedule
from lernomatic.param import lr_common


def get_lr_finder(tr:trainer.Trainer,
                  lr_min:float=1e-7,
                  lr_max:float=1.0,
                  lr_select_method:str='max_acc',
                  num_epochs:int=8,
                  explode_thresh:float=8.0,
                  find_type:str='LogFinder',
                  print_every:int=32) -> lr_common.LRFinder:

    if not hasattr(lr_common, find_type):
        raise ValueError('Unknown learning rate finder type [%s]' % str(find_type))

    lr_find_obj = getattr(lr_common, find_type)
    lr_finder = lr_find_obj(
        tr,
        lr_min           = lr_min,
        lr_max           = lr_max,
        lr_select_method = lr_select_method,
        num_epochs       = num_epochs,
        explode_thresh   = explode_thresh,
        print_every      = print_every
    )

    return lr_finder


def get_lr_finder_binary(tr:trainer.Trainer,
                         lr_min:float,
                         lr_max:float,
                         lr_select_method:str='max_acc',
                         num_epochs:int=8,
                         explode_thresh:float=8.0,
                         print_every:int=32) -> lr_common.LogFinder:
    lr_finder = lr_common.LogFinder(
        tr,
        lr_min           = lr_min,
        lr_max           = lr_max,
        lr_select_method = lr_select_method,
        num_epochs       = num_epochs,
        explode_thresh   = explode_thresh,
        print_every      = print_every
    )

    return lr_finder


def get_scheduler(lr_min:float,
                  lr_max:float,
                  stepsize:int = 5000,
                  sched_type='TriangularScheduler') -> schedule.LRScheduler:
    if sched_type is None:
        return None

    if not hasattr(schedule, sched_type):
        raise ValueError('Unknown scheduler type [%s]' % str(sched_type))

    lr_sched_obj = getattr(schedule, sched_type)
    lr_scheduler = lr_sched_obj(
        stepsize = stepsize,
        lr_min = lr_min,
        lr_max = lr_max
    )

    return lr_scheduler
