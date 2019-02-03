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

    def get_lr(self, batch_idx, loss):

        self.avg_loss = self.beta * self.avg_loss + (1 - self.beta) * loss
        smoothed_loss = self.avg_loss / (1 - self.beta ** batch_num+1)

        if (batch_num > self.min_batches) and (smoothed_loss > (self.best_factor * self.best_loss)):
            return self.learning_rate, True

        if (smoothed_loss < self.best_loss) or (batch_num == 0):
            self.best_loss = smoothed_loss
            print('\t best loss is %f [lr = %f, smoothed loss = %f]' %\
                  (self.best_loss, self.learning_rate, smoothed_loss))

        stepsize = self.stepsize_epoch * self.num_batches
        cycle = np.floor(1 + self.num_epochs / (2 * stepsize))
        x = np.abs(self.num_epochs / stepsize - (2 * cycle + 1))
        self.learning_rate = self.lr_min + (self.lr_max - self.lr_min) *\
            np.maximum(0.0, (1.0 - np.abs(x)))

        log_lr = np.log10(self.learning_rate)
        # save history
        self.loss_history.append(loss)
        self.log_lr_history.append(log_lr)
        self.smooth_loss_history.append(smoothed_loss)
        self.learning_rate *= self.lr_mult

        return self.learning_rate, False


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


# For now we call the experimental lr_finder LRSearch
class LRSearcher(object):
    """
    LRSearcher
    'Search' for a good learning rate
    """
    def __init__(self, trainer, **kwargs):
        self.trainer = trainer
        # learning params
        self.num_epochs     = kwargs.pop('num_epochs', 8)
        # lr params
        self.lr_mult        = kwargs.pop('lr_mult', 0)
        self.lr_min         = kwargs.pop('lr_min', 1e-6)
        self.lr_max         = kwargs.pop('lr_max', 10)
        self.explode_thresh = kwargs.pop('explode_thresh', 4)      # fast.ai uses 4 * min_smoothed_loss
        self.beta           = kwargs.pop('beta', 0.999)
        self.gamma          = kwargs.pop('gamma', 0.999995)
        # other
        self.print_every    = kwargs.pop('print_every', 20)

        # trainer and model params
        self.model_params = None
        self.trainer_params = None
        # loss params
        self.avg_loss = 0.0
        self.best_loss = 1e6
        # learning rate params
        self.learning_rate = 0.0

        self._init_history()

    def __repr__(self):
        return 'LRSearcher'

    def _print_find(self, epoch, batch_idx, loss):
        print('[FIND_LR] :  Epoch    iteration          loss    best loss (smooth)  lr')
        print('            [%d/%d]     [%6d/%6d]    %.6f     %.6f     %.6f' %\
            (epoch, self.num_epochs, batch_idx, len(self.trainer.train_loader),
            loss, self.best_loss, self.learning_rate)
        )

    def _init_history(self):
        self.smooth_loss_history = []
        self.log_lr_history = []
        self.acc_history = []

    def _calc_loss(self, batch_idx, loss):
        self.avg_loss = self.beta * self.avg_loss + (1.0 - self.beta) * loss
        smooth_loss = self.avg_loss / (1.0 - self.beta ** (batch_idx+1))

        return smooth_loss

    def save_model_params(self, params):
        self.model_params = copy.deepcopy(params)

    def save_trainer_params(self, params):
        self.trainer_params = copy.deepcopy(params)

    def load_model_params(self):
        return self.model_params

    def load_trainer_params(self):
        return self.trainer_params

    def find_lr(self):
        raise NotImplementedError('This method should be implemented in subclass')


class LogSearcher(LRSearcher):
    """
    Implements linear learning rate search
    """
    def __init__(self, trainer, **kwargs):
        super(LogSearcher, self).__init__(trainer, **kwargs)

    def __repr__(self):
        return 'LogSearcher'

    def __str__(self):
        s = []
        s.append('LogSearcher. lr range [%f -> %f]\n' % (self.lr_min, self.lr_max))
        return ''.join(s)

    def find_lr(self):
        if self.trainer.train_loader is None:
            raise ValueError('No train_loader in trainer')
        if self.trainer.test_loader is None:
            raise ValueError('No test_loader in trainer')

        self.save_model_params(self.trainer.get_model_params())
        self.save_trainer_params(self.trainer.get_trainer_params())
        self.learning_rate = self.lr_min
        self.lr_mult = (self.lr_max / self.lr_min) ** (1.0 / len(self.trainer.train_loader))

        # train the network for
        explode = False
        for epoch in range(self.num_epochs):
            for batch_idx, (data, labels) in enumerate(self.trainer.train_loader):
                data = data.to(self.trainer.device)
                labels = labels.to(self.trainer.device)

                self.trainer.optimizer.zero_grad()
                outputs = self.trainer.model(data)
                loss = self.trainer.criterion(outputs, labels)

                #smooth_loss = self._calc_loss(batch_idx, loss.item())
                self.avg_loss = self.beta * self.avg_loss + (1.0 - self.beta) * loss.item()
                smooth_loss = self.avg_loss / (1.0 - self.beta ** (batch_idx+1))
                if smooth_loss < self.best_loss:
                    self.best_loss = smooth_loss

                if batch_idx % self.print_every == 0:
                    self._print_find(epoch, batch_idx, loss.item())

                if smooth_loss > self.explode_thresh * self.best_loss:
                    explode = True
                    break

                # update history
                self.smooth_loss_history.append(smooth_loss)
                self.log_lr_history.append(np.log10(self.learning_rate))
                loss.backward()
                self.trainer.optimizer.step()

                # update learning rate
                self.learning_rate *= self.lr_mult
                self.trainer.set_learning_rate(self.learning_rate)

            if explode is True:
                break

        # restore state
        print('[FIND_LR] : restoring trainer state')
        self.trainer.set_trainer_params(self.load_trainer_params())
        print('[FIND_LR] : restoring model state')
        self.trainer.model.state_dict(self.load_model_params())


