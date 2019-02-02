"""
LEARNING_RATE
Tools for finding optimal learning rate

Stefan Wong 2018
"""

import numpy as np
import copy

"""
TODO : stuff that needs to be implemented. What we want is to do some
mock training between some evenly (or perhaps logarithmically) spaced sets
learning rates. Each time we 'train' for a fixed number of iterations (if
we select 0, then we train for one epoch)
"""

# debug
#from pudb import set_trace; set_trace()


class LRFinder(object):
    """
    LRFinder
    Base class for learning rate finders
    """
    def __init__(self, batches_per_epoch, **kwargs):
        # get keyword args
        self.verbose        = kwargs.pop('verbose', False)
        self.max_iter       = kwargs.pop('max_iter', 5000)
        self.stepsize_epoch = kwargs.pop('stepsize_epoch', 8)
        self.best_factor    = kwargs.pop('best_factor', 8.0)
        self.min_batches    = kwargs.pop('min_batches', 80)
        self.num_epochs     = kwargs.pop('num_epochs', 8)
        # lr params
        self.lr_mult        = kwargs.pop('lr_mult', 0)
        self.lr_min         = kwargs.pop('lr_min', 1e-6)
        self.lr_max         = kwargs.pop('lr_max', 10)
        self.lr_thresh      = kwargs.pop('lr_thresh', 4)      # fast.ai uses 4 * min_smoothed_loss
        self.beta           = kwargs.pop('beta', 0.999)
        self.gamma          = kwargs.pop('gamma', 0.999995)
        # for setting lr_mult
        self.num_batches = batches_per_epoch

        # store original model and trainer params
        self.model_params = None
        self.trainer_params = None
        # init losses
        self.avg_loss = 0.0
        self.best_loss = 0.0
        self.learning_rate = self.lr_min

        self._init_history()
        self._init_lr_mult()

    def __repr__(self):
        return 'LRFinder'

    def __str__(self):
        s = []
        s.append('LRFinder (%.3f -> %.3f)\n' % (self.lr_min, self.lr_max))
        return ''.join(s)

    def _init_history(self):
        # TODO : in future try to pre-allocate these...
        self.lr_history          = []
        self.loss_history        = []
        self.log_lr_history      = []
        self.smooth_loss_history = []

    def _init_lr_mult(self):
        if self.lr_mult == 0:
            self.lr_mult = (self.lr_max / self.lr_min) ** (1 / self.num_batches)

    def _get_params(self):
        params = dict()
        params['lr_min']     = self.lr_min
        params['lr_max']     = self.lr_max
        params['lr_thresh']  = self.lr_thresh
        params['beta']       = self.beta
        params['gamma']      = self.gamma
        params['num_epochs'] = self.num_epochs
        params['verbose']    = self.verbose

        return params

    def _set_params(self, params):
        self.lr_min     = params['lr_min']
        self.lr_max     = params['lr_max']
        self.lr_thresh  = params['lr_thresh']
        self.beta       = params['beta']
        self.gamma      = params['gamma']
        self.num_epochs = params['num_epochs']
        self.verbose    = params['verbose']

    def cache_model_params(self, params):
        self.model_params = copy.deepcopy(params)

    def cache_trainer_params(self, params):
        self.trainer_params = copy.deepcopy(params)

    def get_model_params(self):
        return self.model_params

    def get_trainer_params(self):
        return self.trainer_params

    def get_best_lr(self):
        return self.best_loss

    def set_dataset_params(self, dataset_size, batch_size):
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def get_lr(self, dataset_size, batch_size):
        raise NotImplementedError


class LinearFinder(LRFinder):
    def __init__(self, batches_per_epoch, **kwargs):
        super(LinearFinder, self).__init__(batches_per_epoch, **kwargs)

    def __repr__(self):
        return 'LinearFinder'

    def __str__(self):
        s = []
        s.append('LinearFinder [%.3f -> %.3f]\n' % (self.lr_min, self.lr_max))
        return ''.join(s)

    def get_lr(self, batch_num, loss):
        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** batch_num+1)

        if (batch_num > self.min_batches) and (smoothed_loss > (self.best_factor * self.best_loss)):
            return self.learning_rate, True

        if (smoothed_loss < self.best_loss) or (batch_num == 0):
            self.best_loss = smoothed_loss
            print('\t best loss is %f [lr = %f, smoothed loss = %f]' %\
                  (self.best_loss, self.learning_rate, smoothed_loss))
        log_lr = np.log10(self.learning_rate)
        # save history
        self.loss_history.append(loss)
        self.log_lr_history.append(log_lr)
        self.smooth_loss_history.append(smoothed_loss)
        self.learning_rate *= self.lr_mult

        # debug
        #print('[LinearFinder] : log_lr : %.4f, smooth loss : %f, best loss : %f' %\
        #      (log_lr, smoothed_loss, self.best_loss))

        return self.learning_rate, False


class TriangularFinder(LRFinder):
    """
    TriangularFinder
    Find learning rate using traingular annealing
    """
    def __init__(self, batches_per_epoch, **kwargs):
        super(TriangularFinder, self).__init__(batches_per_epoch, **kwargs)

    def __repr__(self):
        return 'TriangularFinder'

    def __str__(self):
        s = []
        s.append('TriangularFinder [%.3f -> %.3f]\n' % (self.lr_min, self.lr_max))
        return ''.join(s)

    def old_get_lr(self, dataset_size, batch_size):
        stepsize = self.stepsize_epoch * int(dataset_size / batch_size)
        cycle = np.floor(1 + self.num_epochs / (2 * stepsize))
        x = np.abs(self.num_epochs / stepsize - (2 * cycle + 1))
        rate = self.lr_min + (self.lr_max - self.lr_min) *\
            np.maximum(0, (1 - np.abs(x)))

        return rate

    def get_lr(self, batch_idx, loss):
        # compute average loss here
        pass


class Triangular2Finder(LRFinder):
    """
    Triangular2Finder
    Find learning rate using triangular2 annealing
    """
    def __init__(self, batches_per_epoch, **kwargs):
        super(Triangular2Finder, self).__init__(batches_per_epoch, **kwargs)

    def __repr__(self):
        return 'Triangular2Finder'

    def __str__(self):
        s = []
        s.append('Triangular2Finder [%.3f -> %.3f]\n' % (self.lr_min, self.lr_max))
        return ''.join(s)

    def old_get_lr(self, dataset_size, batch_size):
        stepsize = self.stepsize_epoch * int(dataset_size / batch_size)
        cycle = np.floor(1 + self.num_epochs / (2 * stepsize))
        x = np.abs(self.num_epochs / stepsize - (2 * cycle + 1))
        rate = self.lr_min + (self.lr_max - self.lr_min) *\
            np.minimum(1, np.maximum(0, (1 - np.abs(x)) / np.pow(2, cycle)))

        return rate


class CosFinder(LRFinder):
    """
    CosFinder
    Find learning rate using Cosine annealing
    """
    def __init__(self, batches_per_epoch, **kwargs):
        super(CosFinder, self).__init__(batches_per_epoch, **kwargs)

    def __repr__(self):
        return 'CosFinder'
