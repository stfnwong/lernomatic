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
    def __init__(self, **kwargs) -> None:
        # learning rate keyword args
        self.lr_min     : float = kwargs.pop('lr_min', 1e-4)
        self.lr_max     : float = kwargs.pop('lr_max', 1.0)
        self.stepsize   : int   = kwargs.pop('stepsize', 1000)
        self.start_iter : float = kwargs.pop('start_iter', 0)

        self.no_lr_history   : bool = kwargs.pop('no_lr_history', False)
        self.lr_history_size : int  = kwargs.pop('lr_history_size', 0)
        self.lr_history_ptr  = 0
        if self.no_lr_history is True:
            self.lr_history_size = 0
        self._init_history()

    def __repr__(self) -> str:
        return 'LRScheduler'

    def __str__(self) -> str:
        s = []
        s.append('LRScheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def _init_history(self) -> None:
        if self.lr_history_size > 0:
            self.lr_history = np.zeros(self.lr_history_size)
        else:
            self.lr_history = None

    def _update_lr_history(self, lr: float) -> None:
        if self.lr_history is None:
            return
        self.lr_history[self.lr_history_ptr] = lr
        self.lr_history_ptr += 1

    def plot_history(self, ax, title=None) -> None:
        if self.lr_history_ptr == 0 or self.lr_history is None:
            raise ValueError('No history recorded in %s' % repr(self))

        ax.plot(np.arange(self.lr_history_ptr), self.lr_history[0 : self.lr_history_ptr])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning rate')
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('[%s] learning rate history' % repr(self))

    def get_lr(self, cur_iter: int) -> float:
        raise NotImplementedError('This should be implemented in the derived class')


class StepScheduler(LRScheduler):
    """
    StepScheduler
    Implements stepped learning rate annealing scheme . Steps down from
    lr_max to lr_min in steps of lr_decay every lr_decay_every

    Arguments:
        lr_decay       - Amount to decay learning rate by at each step
        lr_decay_every - Number of iterations to wait between decay steps

    """
    def __init__(self, **kwargs) -> None:
        self.lr_decay       : float = kwargs.pop('lr_decay',  0.001)
        self.lr_decay_every : float = kwargs.pop('lr_decay_every', 10000)   # unit is iterations
        super(StepScheduler, self).__init__(**kwargs)
        self.cur_lr         : float = self.lr_max

    def __repr__(self) -> str:
        return 'StepScheduler'

    def __str__(self) -> str:
        s = []
        s.append('Step-wise Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter: int) -> float:
        if cur_iter % self.lr_decay_every == 0:
            new_lr = self.cur_lr * self.lr_decay
            self.cur_lr = new_lr
            self._update_lr_history(new_lr)
            return new_lr

        self._update_lr_history(self.cur_lr)
        return self.cur_lr


class LinearDecayScheduler(LRScheduler):
    """
    LinearDecayScheduler
    Decays the learning rate linearly over a fixed number of iterations

    Arguments:
        num_iters - Number of iteratons across which to apply the decay
                    schedule. After this number of iterations have elapsed
                    the learning rate will remain fixed at lr_min

    """
    def __init__(self, **kwargs) -> None:
        self.num_iters :int = kwargs.pop('num_iters', 100000)
        super(LinearDecayScheduler, self).__init__(**kwargs, lr_history_size = self.num_batches)

        self.lr_mult = (self.lr_max - self.lr_min) / self.num_batches
        self.cur_lr = self.lr_max

    def __repr__(self) -> str:
        return 'LinearDecayScheduler'

    def __str__(self) -> str:
        s = []
        s.append('Linear Decay Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter: int) -> float:
        if cur_iter < self.num_iters:
            new_lr = self.cur_lr * self.lr_mult
        else:
            new_lr = self.cur_lr
        self._update_lr_history(new_lr)

        return new_lr


class LogDecayScheduler(LRScheduler):
    def __init__(self, **kwargs) -> None:
        self.num_iter = kwargs.pop('num_iter', 50000)
        super(LogDecayScheduler, self).__init__(**kwargs)
        self.cur_lr = self.lr_max

        self.lr_mult = (self.lr_max - self.lr_min) ** (1.0 / self.num_iter)

    def __repr__(self) -> str:
        return 'LogDecayScheduler'

    def __str__(self) -> str:
        s = []
        s.append('Logarithmic Decay Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter: int) -> float:
        if cur_iter < self.num_iter:
            new_lr = self.cur_lr * self.lr_mult
        else:
            new_lr = self.cur_lr
        self._update_lr_history(new_lr)

        return new_lr


class ExponentialDecayScheduler(LRScheduler):
    def __init__(self, **kwargs) -> None:
        self.k = kwargs.pop('k', 0.001)
        super(ExponentialDecayScheduler, self).__init__(**kwargs)
        self.cur_lr = self.lr_max

    def __repr__(self) -> str:
        return 'ExponentialDecayScheduler'

    def __str__(self) -> str:
        s = []
        s.append('Exponential Decay Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter: int) -> float:
        new_lr = self.lr_max * np.exp(-self.k * cur_iter)
        self._update_lr_history(new_lr)

        return new_lr


class WarmRestartScheduler(LRScheduler):
    def __init__(self, **kwargs) -> None:
        self.step_per_epoch = kwargs.pop('step_per_epoch', 1000)
        super(WarmRestartScheduler, self).__init__(**kwargs)
        self.cur_lr = self.lr_max
        self.batch_since_restart = 0

    def __repr__(self) -> str:
        return 'WarmRestartScheduler'

    def __str__(self) -> str:
        s = []
        s.append('Warm Restart Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter: int) -> float:
        if cur_iter < 1:
            restart_frac = 1.0 / (2 * self.stepsize * self.step_per_epoch)
        else:
            restart_frac = self.batch_since_restart / (2 * self.stepsize * self.step_per_epoch)
        new_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(restart_frac * np.pi))
        self._update_lr_history(new_lr)

        return new_lr


class TriangularScheduler(LRScheduler):
    """
    TriangularScheduler
    Implements the triangular CLR learning rate schedule from
    'Cyclical Learning Rates for Training Neural Networks' (L. N. Smith, 2017)
    """
    def __init__(self, **kwargs) -> None:
        super(TriangularScheduler, self).__init__(**kwargs)

    def __repr__(self) -> str:
        return 'TriangularScheduler'

    def __str__(self) -> str:
        s = []
        s.append('Triangular Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter: int) -> float:
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


class TriangularExpScheduler(LRScheduler):
    def __init__(self, **kwargs) -> None:
        self.k :float = kwargs.pop('k', 0.1)
        super(TriangularExpScheduler, self).__init__(**kwargs)

    def __repr__(self) -> str:
        return 'TriangularExpScheduler'

    def __str__(self) -> str:
        s = []
        s.append('Triangular Learning Rate Scheduler (exponential decay)\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter: int) -> float:
        itr = cur_iter - self.start_iter
        if itr > 0:
            cycle = itr / (2 * self.stepsize)
            x = itr - (2 * cycle + 1) * self.stepsize
            x /= self.stepsize
            rate = self.lr_min + (self.lr_max - self.lr_min) *\
                np.maximum(0.0, 1.0 - np.abs(x))
            rate = rate * np.exp(-self.k * cur_iter)
            self._update_lr_history(rate)
            return rate

        self._update_lr_history(self.lr_min)
        return self.lr_min


class Triangular2Scheduler(LRScheduler):
    def __init__(self, **kwargs) -> None:
        super(Triangular2Scheduler, self).__init__(**kwargs)

    def __repr__(self) -> str:
        return 'Triangular2Scheduler'

    def __str__(self) -> str:
        s = []
        s.append('Triangular2 Learning Rate Scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter: int) -> float:
        itr = cur_iter - self.start_iter
        if itr > 0:
            cycle = itr / (2 * self.stepsize)
            x = itr - (2 * cycle + 1) * self.stepsize
            x /= self.stepsize
            rate = self.lr_min + (self.lr_max - self.lr_min) *\
                np.minimum(1.0, np.maximum(0.0, (1.0 - np.abs(x) / np.power(2, cycle))))
            self._update_lr_history(rate)
            return rate

        self._update_lr_history(self.lr_min)
        return self.lr_min


class Triangular2ExpScheduler(LRScheduler):
    def __init__(self, **kwargs) -> None:
        self.k :float = kwargs.pop('k', 0.1)
        super(Triangular2ExpScheduler, self).__init__(**kwargs)

    def __repr__(self) -> str:
        return 'Triangular2Expcheduler'

    def __str__(self) -> str:
        s = []
        s.append('Triangular2 Learning Rate Scheduler (exponential decay)\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))

        return ''.join(s)

    def get_lr(self, cur_iter: int) -> float:
        itr = cur_iter - self.start_iter
        if itr > 0:
            cycle = itr / (2 * self.stepsize)
            x = itr - (2 * cycle + 1) * self.stepsize
            x /= self.stepsize
            rate = self.lr_min + (self.lr_max - self.lr_min) *\
                np.minimum(1.0, np.maximum(0.0, (1.0 - np.abs(x) / np.power(2, cycle))))
            rate = rate * np.exp(-self.k * cur_iter)
            self._update_lr_history(rate)
            return rate

        self._update_lr_history(self.lr_min)
        return self.lr_min


class EpochSetScheduler(LRScheduler):
    def __init__(self, schedule: dict, **kwargs) -> None:
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
        if type(schedule) is not dict:
            raise ValueError('schedule must be a dict of epochs and values')
        self.schedule      : dict  = schedule
        self.lr_value      : bool  = kwargs.pop('lr_value', False)       # if true, set the lr to the value in the dict at epoch E
        self.learning_rate : float = 0.0
        super(EpochSetScheduler, self).__init__(**kwargs)

        self._check_schedule()

    def __repr__(self) -> str:
        return 'EpochSetScheduler'

    def __str__(self) -> str:
        s = []
        s.append('EpochSetScheduler learning rate scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))
        if self.lr_value is True:
            s.append(' epoch      : learning rate (set) \n')
        else:
            s.append(' epoch      : learning rate (multiplied by) \n')
        for k, v in self.schedule.items():
            s.append('\t %6d :  %f\n' % (int(k), v))
        s.append('\n')

        return ''.join(s)

    def _check_schedule(self) -> None:
        if 0 not in self.schedule.keys():
            raise ValueError('Failed to find 0 key in schedule')

        for k in self.schedule.keys():
            if type(k) is not int:
                raise ValueError('Key [%s] is not an integer. Schedule keys must be integers' %\
                                 str(k)
                )

    def get_lr(self, cur_epoch: int) -> float:
        if cur_epoch in self.schedule.keys():
            self.learning_rate = self.schedule[cur_epoch]
        self._update_lr_history(self.learning_rate)

        return self.learning_rate


class TriangularDecayWhenAcc(LRScheduler):
    def __init__(self, **kwargs) -> None:
        self.lr_decay   : float = kwargs.pop('lr_decay', 0.9)
        self.acc_thresh : float = kwargs.pop('acc_thresh', 0.05)
        self.best_acc   : float = 0.0
        super(TriangularDecayWhenAcc, self).__init__(**kwargs)

    def __repr__(self) -> str:
        return 'TriangularDecayWhenAcc'

    def __str__(self) -> str:
        s = []
        s.append('Triangular Learning Rate Scheduler with Accuracy Decay\n')
        s.append('lr range %.4f -> %.4f, lr decay : %.4f \n' % (self.lr_min, self.lr_max, self.lr_decay))

        return ''.join(s)

    def get_lr(self, cur_iter: int, cur_acc: int) -> float:
        if cur_acc > self.best_acc:
            self.best_acc = cur_acc
        itr = cur_iter - self.start_iter
        if itr > 0:
            cycle = itr / (2 * self.stepsize)
            x = itr - (2 * cycle + 1) * self.stepsize
            x /= self.stepsize
            rate = self.lr_min + (self.lr_max - self.lr_min) *\
                np.maximum(0.0, 1.0 - np.abs(x))
            if cur_acc < (self.best_acc * (1.0 - self.acc_thresh)):
                rate = rate * self.lr_decay
            self._update_lr_history(rate)

            return rate

        self._update_lr_history(self.lr_min)
        return self.lr_min


class DecayWhenAcc(LRScheduler):
    def __init__(self, **kwargs) -> None:
        self.initial_lr    : float = kwargs.pop('initial_lr', 0.01)
        self.acc_thresh    : float = kwargs.pop('acc_thresh', 0.05)
        self.lr_decay      : float = kwargs.pop('lr_decay', 0.9)
        super(DecayWhenAcc, self).__init__(**kwargs)
        self.best_acc      : float = 0.0
        self.learning_rate : float = self.initial_lr

    def __repr__(self) -> str:
        return 'DecayWhenAcc'

    def __str__(self) -> str:
        s = []
        s.append('DecayWhenAcc learning rate scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))
        s.append('acc thresh : %.4f\n' % self.acc_thresh)
        s.append('\n')

        return ''.join(s)

    def get_lr(self, cur_acc: int) -> float:
        if cur_acc > self.best_acc:
            self.best_acc = cur_acc
        if cur_acc < (self.best_acc * (1.0 - self.acc_thresh)):
            self.learning_rate = self.learning_rate * self.lr_decay
        self._update_lr_history(self.learning_rate)

        return self.learning_rate


class DecayWhenEpoch(LRScheduler):
    """
    DecayWhenEpoch
    Decay the learning rate every num_epochs by lr_decay
    """
    def __init__(self, **kwargs) -> None:
        self.num_epochs : int   = kwargs.pop('num_epochs', 8)
        self.lr_decay   : float = kwargs.pop('lr_decay', 0.9)
        super(DecayWhenEpoch, self).__init__(**kwargs)

    def __repr__(self) -> str:
        return 'DecayWhenEpoch'

    def __str__(self) -> str:
        s = []
        s.append('DecayWhenEpoch learning rate scheduler\n')
        s.append('lr range %.4f -> %.4f \n' % (self.lr_min, self.lr_max))
        s.append('epoch : %.4f, lr decay : %.4f\n' % (self.num_epochs, self.lr_decay))
        s.append('\n')

    def get_lr(self, cur_epoch: int) -> float:
        if (cur_epoch % self.num_epochs) == 0:
            self.learning_rate = self.learning_rate * self.lr_decay

        return self.learning_rate


# ---- Momentum ----- #
class MtmScheduler(object):
    def __init__(self, **kwargs):
        self.mtm_min = kwargs.pop('mtm_min', 1e-3)
        self.mtm_max = kwargs.pop('mtm_max', 0.9)

    def __repr__(self):
        return 'MtmScheduler'


class TriangularMtmScheduler(MtmScheduler):
    def __init__(self, **kwargs):
        super(TriangularMtmScheduler, self).__init__(**kwargs)

    def get_mtm(self):
        pass
