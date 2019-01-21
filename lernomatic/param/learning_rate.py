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
        self.lr_space     = kwargs.pop('lr_space', 'linear')
        # learning rate info
        self.best_lr      = 0.0
        self.cur_lr       = 0.0
        #self.lr_cache = list()

        # TODO : possibly allow the ability to cache the history of all losses
        # and results...

    def __repr__(self):
        return 'LRFinder'

    def __str__(self):
        s = []
        s.append('LRFinder (%.3f -> %.3f)\n' % (self.start_lr, self.end_lr))
        s.append('%d attempts with %d iters per attempt\n' % (self.max_attempts, self.num_iter))
        return ''.join(s)

    def find_lr(self, print_every=0):
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


    # TODO : save and load parameters
    # TODO : save and load object?
