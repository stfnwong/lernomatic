"""
LEARNING_RATE
Tools for finding optimal learning rate

Stefan Wong 2018
"""

import numpy as np
import copy
import torch

# debug
from pudb import set_trace; set_trace()

class LRFinder(object):
    """
    LRFinder
    Finds optimal learning rates
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
        # gradient params
        self.grad_thresh    = kwargs.pop('grad_thresh', 0.002)
        # other
        self.acc_test       = kwargs.pop('acc_test', False)
        self.print_every    = kwargs.pop('print_every', 20)
        self.verbose        = kwargs.pop('verbose', False)

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
        return 'LRFinder'

    def _print_find(self, epoch, batch_idx, loss):
        print('[FIND_LR] :  Epoch    iteration         loss    best loss (smooth)  lr')
        print('            [%d/%d]     [%6d/%6d]    %.6f     %.6f     %.6f' %\
            (epoch, self.num_epochs, batch_idx, len(self.trainer.train_loader),
            loss, self.best_loss, self.learning_rate)
        )

    def _print_acc(self, avg_test_loss, correct, acc, dataset_size):
        print('[FIND_LR ACC]  : Avg. Test Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
              (avg_test_loss, correct, dataset_size, 100.0 * acc)
        )

    def _init_history(self):
        self.smooth_loss_history = []
        self.log_lr_history = []
        self.acc_history = []
        self.loss_grad_history = []

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

    def check_loaders(self):
        if self.trainer.train_loader is None:
            raise ValueError('No train_loader in trainer')
        if self.trainer.test_loader is None:
            raise ValueError('No test_loader in trainer')

    def find_lr(self):
        raise NotImplementedError('This method should be implemented in subclass')


class LogFinder(LRFinder):
    """
    Implements logarithmic learning rate search
    """
    def __init__(self, trainer, **kwargs):
        super(LogFinder, self).__init__(trainer, **kwargs)

    def __repr__(self):
        return 'LogFinder'

    def __str__(self):
        s = []
        s.append('LogFinder. lr range [%f -> %f]\n' % (self.lr_min, self.lr_max))
        return ''.join(s)

    def find(self):
        """
        find_lr()
        Search for an optimal learning rate
        """
        self.check_loaders()
        # cache parameters for later
        self.save_model_params(self.trainer.get_model_params())
        self.save_trainer_params(self.trainer.get_trainer_params())
        self.learning_rate = self.lr_min
        self.lr_mult = (self.lr_max / self.lr_min) ** (1.0 / len(self.trainer.train_loader))

        self.prev_smooth_loss = 0.0

        if self.verbose:
            print('Finding lr using trainer :')
            print(self.trainer)

        # train the network while varying the learning rate
        explode = False
        for epoch in range(self.num_epochs):
            for batch_idx, (data, labels) in enumerate(self.trainer.train_loader):
                self.trainer.model.train()
                data = data.to(self.trainer.device)
                labels = labels.to(self.trainer.device)

                self.trainer.optimizer.zero_grad()
                outputs = self.trainer.model(data)
                loss = self.trainer.criterion(outputs, labels)

                #smooth_loss = self._calc_loss(batch_idx, loss.item())
                self.prev_avg_loss = self.avg_loss
                self.prev_learning_rate = self.learning_rate
                self.avg_loss = self.beta * self.avg_loss + (1.0 - self.beta) * loss.item()
                smooth_loss = self.avg_loss / (1.0 - self.beta ** (batch_idx+1))

                # gradient
                if batch_idx > 0:
                    loss_grad = self.avg_loss - self.prev_avg_loss
                    self.loss_grad_history.append(loss_grad)

                # save loss
                if smooth_loss < self.best_loss:
                    self.best_loss = smooth_loss

                # display
                if batch_idx % self.print_every == 0:
                    self._print_find(epoch, batch_idx, loss.item())

                # accuracy test
                if self.acc_test is True:
                    if self.trainer.test_loader is not None:
                        self.acc_test(self.trainer.test_loader, batch_idx)
                    else:
                        self.acc_test(self.trainer.train_loader, batch_idx)

                if smooth_loss > self.explode_thresh * self.best_loss:
                    explode = True
                    print('[FIND_LR] loss hit explode threshold [%.3f x best (%f)]' %\
                          (self.explode_thresh, self.best_loss)
                    )
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

    def find_range(self):
        return NotImplementedError

    def acc_test(self, data_loader, batch_idx):
        """
        acc_test()
        Collect accuracy stats while finding learning rate
        """
        test_loss = 0.0
        correct = 0
        self.trainer.model.eval()
        for n, (data, labels) in enumerate(data_loader):
            data = data.to(self.trainer.device)
            labels = labels.to(self.trainer.device)

            with torch.no_grad():
                output = self.trainer.model(data)
            loss = self.trainer.criterion(output, labels)
            test_loss += loss.item()

            # accuracy
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum().item()

            #if (n % self.print_every) == 0:
            #    print('[FIND_LR ACC]  :   iteration         Test Loss')
            #    print('                  [%6d/%6d]  %.6f' %\
            #          (n, len(data_loader), loss.item()))

        avg_test_loss = test_loss / len(data_loader)
        acc = correct / len(data_loader.dataset)
        self.acc_history.append(acc)
        if batch_idx % self.print_every == 0:
            print('[FIND_LR ACC]  : Avg. Test Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
                (avg_test_loss, correct, len(data_loader.dataset),
                100.0 * acc)
            )
