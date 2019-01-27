"""
LEARNING_RATE
Tools for finding optimal learning rate

Stefan Wong 2018
"""

import numpy as np

"""
TODO : stuff that needs to be implemented. What we want is to do some
mock training between some evenly (or perhaps logarithmically) spaced sets
learning rates. Each time we 'train' for a fixed number of iterations (if
we select 0, then we train for one epoch)
"""

# debug
from pudb import set_trace; set_trace()

# Basically a typed dict()
class LRParams(object):
    def __init__(self):
        self.lr_list = list()
        self.best_lr = 0.0
        self.mom_list = list()
        self.best_mom = 0.0


class LRFinder(object):
    def __init__(self, trainer, **kwargs):
        self.trainer = trainer
        # get keyword args
        self.verbose      = kwargs.pop('verbose', False)
        self.max_attempts = kwargs.pop('max_attempts', 5000)
        self.num_iter     = kwargs.pop('num_iter', 1000)    # number of steps to perform training
        self.start_lr     = kwargs.pop('start_lr', 1e-6)
        self.end_lr       = kwargs.pop('end_lr', 10)
        self.epoch_mult   = kwargs.pop('epoch_mult', 8)
        self.lr_mult      = kwargs.pop('lr_mult', 1.05)
        self.gamma        = kwargs.pop('gamma', 0.999995)
        # lr params
        self.lr_thresh    = kwargs.pop('lr_thresh', 4)      # fast.ai uses 4 * min_smoothed_loss
        self.lr_beta      = kwargs.pop('lr_beta', 0.999)

    def __repr__(self):
        return 'LRFinder'

    def __str__(self):
        s = []
        s.append('LRFinder (%.3f -> %.3f)\n' % (self.start_lr, self.end_lr))
        return ''.join(s)

    def get_stepsize(self):
        return self.epoch_mult * int(len(self.trainer.train_loader.dataset) / self.trainer.batch_size)

    def get_cycle(self, stepsize):
        return np.floor(1 + self.trainer.num_epochs / (2 * stepsize))

    def get_lr(self, cycle, stepsize):
        x = np.abs(self.trainer.num_epochs / stepsize - 2 * cycle + 1)
        return self.start_lr + (self.end_lr - self.start_lr) * np.max(0, (1 - x))

    def range_test(self, test_num_epochs, lr_low, lr_high, lr_step):
        # Cache the original number of epochs
        self.trainer_num_epochs = self.trainer.num_epochs
        self.trainer.num_epochs = test_num_epochs

        epoch_num_iter = len(self.trainer.train_loader)
        lrs = np.linspace(lr_low, lr_high, lr_step)


    def old_find_lr(self, print_every=0):
        # Prepare a loader
        loader_iterator = iter(self.trainer.train_loader)

        best_loss = 1e8
        for lr_idx, lr_attempt in enumerate(range(self.max_attempts)):
            cand_lr = (np.random.uniform(self.start_lr, self.end_lr))

            print('\t [LRFINDER] :  Attempt [%d / %d] with candidate learning rate %.4f' %\
                  (lr_idx+1, self.max_attempts, cand_lr))

            self.trainer.learning_rate = cand_lr
            self.trainer.num_epochs    = 1
            if print_every > 0:
                self.trainer.print_every = print_every
            # train
            self.trainer.train_fixed(self.num_iter)

            # check how well we did
            # TODO: also need to check, for example, if loss exploded or not,
            # etc
            if self.trainer.loss_history[self.num_iter] < best_loss:
                best_loss = self.trainer.loss_history[self.num_iter]
                self.best_lr = cand_lr
                if self.verbose:
                    print('\t [LRFINDER] : best learning rate is %.3f' % self.best_lr)

            # if we didn't do well enough then reset history and try again
            self.trainer.reset_history()


class TriangularLR(LRFinder):
    def __init__(self, trainer, **kwargs):
        super(TriangularLR, self).__init__(trainer, **kwargs)

    def __repr__(self):
        return 'TriangularLR'

    def __str__(self):
        s = []
        s.append('TriangularLR [%.3f -> %.3f]\n' % (self.start_lr, self.end_lr))
        return ''.join(s)

    def find_lr(self):
        pass
