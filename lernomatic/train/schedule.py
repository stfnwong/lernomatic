"""
SCHEDULE
Modules for parameter scheduling during training

Stefan Wong 2019
"""

import numpy as np

# debug
#from pudb import set_trace; set_trace()

# ---- Learning Rate ---- #
class LRScheduler(object):
    """
    LRScheduler
    Base class for all lernomatic learning rate schedulers
    """
    def __init__(self, **kwargs):
        # learning rate keyword args
        self.lr_min     = kwargs.pop('lr_min', 1e-4)
        self.lr_max     = kwargs.pop('lr_max', 1.0)
        self.stepsize   = kwargs.pop('stepsize', 1000)
        self.start_iter = kwargs.pop('start_iter', 0)

        self.lr_history_size = kwargs.pop('lr_history_size', 0)
        self.lr_history_ptr  = 0
        self._init_history()

    def __repr__(self):
        return 'LRScheduler'

    def __str__(self):      # TODO : something fancy
        return self.__repr__()

    def _init_history(self):
        if self.lr_history_size > 0:
            self.lr_history = np.zeros(self.lr_history_size)
        else:
            self.lr_history = None

    def _update_lr_history(self, lr):
        if self.lr_history is None:
            return

        self.lr_history[self.lr_history_ptr] = lr
        self.lr_history_ptr += 1

    def get_lr(self, cur_iter):
        raise NotImplementedError('This should be implemented in the derived class')


class StepLRScheduler(LRScheduler):
    """
    StepLRScheduler
    Implements stepped learning rate annealing scheme . Steps down from
    lr_max to lr_min in steps of lr_decay every lr_decay_every
    """
    def __init__(self, **kwargs):
        super(StepLRScheduler, self).__init__(**kwargs)
        self.lr_decay       = kwargs.pop('lr_decay',  0.001)
        self.lr_decay_every = kwargs.pop('lr_decay_every', 10000)
        self.cur_lr         = self.lr_max

    def __repr__(self):
        return 'StepLRScheduler'

    def __str__(self):
        s = []
        s.append('Step-wise Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter):
        if cur_iter % self.lr_decay_every == 0:
            new_lr = self.cur_lr * self.lr_decay
            self.cur_lr = new_lr
            self._update_lr_history(new_lr)
            return new_lr

        self._update_lr_history(self.cur_lr)
        return self.cur_lr


# TODO : implement this
class WarmRestartScheduler(LRScheduler):
    def __init__(self, **kwargs):
        super(WarmRestartScheduler, self).__init__(**kwargs)


class TriangularLRScheduler(LRScheduler):
    """
    TriangularLRScheduler
    Implements the triangular CLR learning rate schedule from
    'Cyclical Learning Rates for Training Neural Networks' (L. N. Smith, 2017)
    """
    def __init__(self, **kwargs):
        super(TriangularLRScheduler, self).__init__(**kwargs)

    def __repr__(self):
        return 'TriangularLRScheduler'

    def __str__(self):
        s = []
        s.append('Triangular Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter):
        itr = cur_iter - self.start_iter
        if itr > 0:
            cycle = itr / (2 * self.stepsize)
            x = itr - (2 * cycle + 1) * self.stepsize
            x /= self.stepsize
            rate = self.lr_min + (self.lr_max - self.lr_min) *\
                np.maximum(0.0, 1.0 - np.abs(x))
            self._update_lr_history(rate)
            return rate

        self._update_lr_history(self.lr_min)
        return self.lr_min


class Triangular2LRScheduler(LRScheduler):
    def __init__(self, **kwargs):
        super(Triangular2LRScheduler, self).__init__(**kwargs)

    def __repr__(self):
        return 'Triangular2LRScheduler'

    def __str__(self):
        s = []
        s.append('Triangular2 Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter):
        itr = cur_iter - self.start_iter
        if itr > 0:
            cycle = itr / (2 * self.stepsize)
            x = itr - (2 * cycle + 1) * self.stepsize
            x /= self.stepsize
            rate = self.lr_min + (self.lr_max - self.lr_min) *\
                np.minimum(1.0, np.maximum(0.0, (1.0 - np.abs(x) / np.pow(2, cycle))))
            self._update_lr_history(rate)
            return rate

        self._update_lr_history(self.lr_min)
        return self.lr_min


class EpochSetScheduler(LRScheduler):
    def __init__(self, schedule, **kwargs):
        """
        EpochSetScheduler
        Adjust learning rate on a fixed schedule at the given epochs

        Arguments:
            schedule - A dictionary specifying the schedule to use. Each key must be an
                       integer specifying the epoch at which to apply the value. The value
                       may either be a float by which to multiply the current learning rate,
                       or a value to set the current learning rate to. The value is applied
                       when the training reaches the specified epoch. There must be a '0'
                       key that indicates the initial learning rate.

            lr_value - If true, set the lr to the value in the schedule at the given
                       epoch. If false, multiply the current learning rate by the value
                       in the schedule at the given epoch.
        """
        super(EpochSetScheduler, self).__init__(**kwargs)
        if type(schedule) is not dict:
            raise ValueError('schedule must be a dict of epochs and values')
        self.schedule = schedule
        self.lr_value = kwargs.pop('lr_value', False)       # if true, set the lr to the value in the dict at epoch E
        self.learning_rate = 0.0

        self._check_schedule()

    def __repr__(self):
        return 'EpochSetScheduler'

    def __str__(self):
        s = []
        s.append('EpochSetScheduler Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))
        if self.lr_value is True:
            s.append(' Epoch      : learning rate (set) \n')
        else:
            s.append(' Epoch      : learning rate (multiplied by) \n')
        for k, v in self.schedule.items():
            s.append('\t %6d :  %f\n' % (int(k), v))
        s.append('\n')

        return ''.join(s)

    def _check_schedule(self):
        if 0 not in self.schedule.keys():
            raise ValueError('Failed to find 0 key in schedule')

        for k in self.schedule.keys():
            if type(k) is not int:
                raise ValueError('Key [%s] is not an integer. Schedule keys must be integers' %\
                                 str(k)
                )

    def get_lr(self, cur_epoch):
        if cur_epoch in self.schedule.keys():
            self.learning_rate = self.schedule[cur_epoch]

        return self.learning_rate




# ---- Momentum ----- #
class MScheduler(object):
    def __init__(self, **kwargs):
        self.mom_min = kwargs.pop('mom_min', 1e-3)
        self.mom_max = kwargs.pop('mom_max', 0.9)


class TriangularMScheduler(MScheduler):
    def __init__(self, **kwargs):
        super(TriangularMScheduler, self).__init__(**kwargs)
