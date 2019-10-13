"""
LR_COMMON
Tools for finding optimal learning rate

Stefan Wong 2018
"""

import importlib
import pickle
import numpy as np
import copy
import torch

# KDE tools
from lernomatic.util import kernel_util
from lernomatic.util import math_util

# debug
#from pudb import set_trace; set_trace()


class LRFinder(object):
    """
    LRFinder
    Finds optimal learning rates
    """
    def __init__(self, trainer, **kwargs) -> None:
        valid_select_methods = ('max_acc', 'thresh_acc', 'min_loss', 'max_range', 'kde')
        self.trainer          = trainer
        # learning params
        self.num_epochs       = kwargs.pop('num_epochs', 8)
        # lr params
        self.lr_mult          :float = kwargs.pop('lr_mult', 0.0)
        self.lr_min           :float = kwargs.pop('lr_min', 1e-6)
        self.lr_max           :float = kwargs.pop('lr_max', 1.0)
        self.explode_thresh   :float = kwargs.pop('explode_thresh', 4.0)      # fast.ai uses 4 * min_smoothed_loss
        self.beta             :float = kwargs.pop('beta', 0.999)
        self.gamma            :float = kwargs.pop('gamma', 0.999995)
        self.lr_min_factor    :float = kwargs.pop('lr_min_factor', 2.0)
        self.lr_max_scale     :float = kwargs.pop('lr_max_scale', 1.0)
        self.lr_select_method :str   = kwargs.pop('lr_select_method', 'max_acc')
        self.lr_trunc         :int   = kwargs.pop('lr_trunc', 10)       # how much of lr result to truncate off on either side
        # search time
        self.max_batches      :int   = kwargs.pop('max_batches', 0)
        # gradient params
        self.grad_thresh      :float = kwargs.pop('grad_thresh', 0.002)
        # other
        self.acc_test         :bool  = kwargs.pop('acc_test', True)
        self.print_every      :int   = kwargs.pop('print_every', 20)
        self.verbose          :bool  = kwargs.pop('verbose', False)
        # can add some unique id (eg: for comparing states from different
        # experiments)
        self.expr_id          :str   = kwargs.pop('expr_id', None)

        # trainer and model params
        self.model_params   = None
        self.trainer_params = None
        # loss params
        self.avg_loss       = 0.0
        self.best_loss      = 1e6
        self.best_loss_idx  = 0
        # acc params
        self.best_acc       = 0.0
        self.best_acc_idx   = 0
        # learning rate params
        self.learning_rate  = 0.0

        self._init_history()

    def __repr__(self) -> str:
        return 'LRFinder'

    def __str__(self) -> str:
        s = []
        s.append('%s (%.4f -> %.4f)\n' % (repr(self), self.lr_min, self.lr_max))
        s.append('Method [%s]\n' % str(self.lr_select_method))
        s.append('avg_loss : %f, best_loss: %f, best_acc: %f\n' % (self.avg_loss, self.best_loss, self.best_acc))
        return ''.join(s)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # remove stuff we don't want to save
        del state['trainer']
        del state['trainer_params']
        del state['model_params']
        # add repr and module path
        state['module_name'] = repr(self)
        state['module_path'] = 'lernomatic.param.lr_common'

        return state

    def __setstate__(self, state:dict) -> None:
        self.__dict__.update(state)

    def _print_find(self, epoch: int, batch_idx: int, loss: float) -> None:
        print('[FIND_LR] :  Epoch    iteration         loss    best loss (smooth)  lr')
        print('            [%d/%d]     [%6d/%6d]    %.6f     %.6f     %.6f' %\
            (epoch, self.num_epochs, batch_idx, len(self.trainer.train_loader),
            loss, self.best_loss, self.learning_rate)
        )

    def _print_acc(self, avg_val_loss: float, correct: int, acc: float, dataset_size: int) -> None:
        print('[FIND_LR ACC]  : Avg. Test Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
              (avg_val_loss, correct, dataset_size, 100.0 * acc)
        )

    def _init_history(self) -> None:
        self.smooth_loss_history = []
        self.log_lr_history = []
        self.acc_history = []
        self.loss_grad_history = []

    def _calc_loss(self, batch_idx: int, loss: float) -> float:
        self.avg_loss = self.beta * self.avg_loss + (1.0 - self.beta) * loss
        smooth_loss = self.avg_loss / (1.0 - self.beta ** (batch_idx+1))

        return smooth_loss

    def _max_acc_loss(self) -> tuple:
        lr_max = self.log_lr_history[self.best_acc_idx] * self.lr_max_scale
        lr_min = lr_max * self.lr_min_factor

        return (lr_min, lr_max)

    def _thresh_acc_loss(self) -> tuple:
        acc_history = np.asarray(self.acc_history)
        clip_point = acc_history.max() * 0.85
        idxs = np.argwhere(acc_history > clip_point)

        lr_max = self.log_lr_history[idxs[0][0]]
        lr_min = self.log_lr_history[idxs[-1][0]]

        return (lr_min, lr_max)

    def _kde_loss(self) -> tuple:
        lr_kde = kernel_util.kde(np.asarray(self.acc_history))
        # clip out the relevant region
        lr_kde = lr_kde[self.lr_trunc : len(lr_kde) - self.lr_trunc]
        clip_point = lr_kde.max() * 0.9

        idxs = np.argwhere(lr_kde > clip_point)
        lr_max = self.log_lr_history[idxs[0][0]]
        lr_min = self.log_lr_history[idxs[-1][0]]

        return (lr_min, lr_max)

    def save(self, filename:str) -> None:
        with open(filename, 'wb') as fp:
            pickle.dump(self.__getstate__(), fp)

    def load(self, filename:str) -> None:
        with open(filename, 'rb') as fp:
            state = pickle.load(fp)
        self.__setstate__(state)

    def save_model_params(self, params: dict) -> None:
        self.model_params = copy.deepcopy(params)

    def save_trainer_params(self, params: dict) -> None:
        self.trainer_params = copy.deepcopy(params)

    def load_model_params(self) -> dict:
        return self.model_params

    def load_trainer_params(self) -> dict:
        return self.trainer_params

    def get_params(self) -> dict:
        return self.__getstate__()

    # TODO : should this return a bool for status?
    def check_loaders(self) -> None:
        if self.trainer.train_loader is None:
            raise ValueError('No train_loader in trainer')
        if self.trainer.val_loader is None:
            raise ValueError('No val_loader in trainer')

    def get_lr_history(self) -> list:
        if len(self.log_lr_history) < 1:
            return None
        return self.log_lr_history

    def find(self) -> tuple:
        raise NotImplementedError('This method should be implemented in subclass')

    def get_lr_range(self) -> tuple:
        if self.lr_select_method == 'thresh_acc':
            lr_min, lr_max = self._thresh_acc_loss()
        elif self.lr_select_method == 'max_acc':
            lr_min, lr_max = self._max_acc_loss()
        elif self.lr_select_method == 'min_loss':
            lr_max = self.log_lr_history[self.best_loss_idx] * self.lr_max_scale
            lr_min = lr_max * self.lr_min_factor
        elif self.lr_select_method == 'max_range':
            lr_max = self.log_lr_history[-1] * self.lr_max_scale
            lr_min = lr_max * self.lr_min_factor
        elif self.lr_select_method == 'kde':
            lr_min, lr_max = self._kde_loss()
        else:
            raise ValueError('Invalid range selection method [%s]' % str(self.lr_select_method))

        return (10**lr_min, 10**lr_max)

    # plotting
    def plot_lr_vs_acc(self, ax, title:str=None, log:bool=False) -> None:
        if len(self.log_lr_history) < 1:
            raise RuntimeError('[%s] no learning rate history' % repr(self))
        if len(self.acc_history) < 1:
            raise RuntimeError('[%s] no accuracy history' % repr(self))

        if log is True:
            ax.plot(np.asarray(self.log_lr_history), np.asarray(self.acc_history))
        else:
            ax.plot(10 ** np.asarray(self.log_lr_history), np.asarray(self.acc_history))

        # Add vertical bars showing learning rate ranges
        lr_min, lr_max = self.get_lr_range()
        if lr_min is not None and lr_max is not None:
            if log is True:
                ax.axvline(x=np.log10(lr_min), color='r', label='lr_min')
                ax.axvline(x=np.log10(lr_max), color='r', label='lr_max')
            else:
                ax.axvline(x=10 ** lr_min, color='r', label='lr_min')
                ax.axvline(x=10 ** lr_max, color='r', label='lr_max')

        ax.set_xlabel('Learning rate')
        ax.set_ylabel('Accuracy')
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('Accuracy vs. Learning rate [%s]' % str(self.lr_select_method))

    def plot_lr_vs_loss(self, ax, title:str=None, log:bool=False) -> None:
        if len(self.log_lr_history) < 1:
            raise ValueError('[%s] no learning rate history' % repr(self))
        if len(self.acc_history) < 1:
            raise ValueError('[%s] no accuracy history' % repr(self))

        if log is True:
            ax.plot(np.asarray(self.log_lr_history), np.asarray(self.smooth_loss_history))
        else:
            ax.plot(10 ** np.asarray(self.log_lr_history), np.asarray(self.smooth_loss_history))

        # Add vertical bars showing learning rate ranges
        lr_min, lr_max = self.get_lr_range()
        if lr_min is not None and lr_max is not None:
            if log is True:
                ax.axvline(x=np.log10(lr_min), color='r', label='lr_min')
                ax.axvline(x=np.log10(lr_max), color='r', label='lr_max')
            else:
                ax.axvline(x=10 ** lr_min, color='r', label='lr_min')
                ax.axvline(x=10 ** lr_max, color='r', label='lr_max')

        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Smooth loss')
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title('Learning rate vs. Loss [%s]' % str(self.lr_select_method))


# ======== Finder schedules ========  #
class LogFinder(LRFinder):
    """
    Implements logarithmic learning rate search
    """
    def __init__(self, trainer, **kwargs) -> dict:
        super(LogFinder, self).__init__(trainer, **kwargs)

    def __repr__(self) -> str:
        return 'LogFinder'

    #def __str__(self) -> str:
    #    s = []
    #    s.append('LogFinder. lr range [%f -> %f]\n' % (self.lr_min, self.lr_max))
    #    return ''.join(s)

    def find(self) -> tuple:
        """
        find_lr()
        Search for an optimal learning rate
        """
        self.check_loaders()
        # cache parameters for later
        self.save_model_params(self.trainer.model.get_net_state_dict())
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
                self.trainer.model.set_train()
                data = data.to(self.trainer.device)
                labels = labels.to(self.trainer.device)

                self.trainer.optimizer.zero_grad()
                outputs = self.trainer.model.forward(data)
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
                    self.best_loss_idx = len(self.log_lr_history)

                # display
                if batch_idx % self.print_every == 0:
                    self._print_find(epoch, batch_idx, loss.item())

                # accuracy test
                if self.acc_test is True:
                    if self.trainer.val_loader is not None:
                        self.acc(self.trainer.val_loader, batch_idx)
                    else:
                        self.acc(self.trainer.train_loader, batch_idx)
                    # keep a record of the best acc
                    if self.acc_history[-1] > self.best_acc:
                        self.best_acc = self.acc_history[-1]
                        self.best_acc_idx = len(self.acc_history)

                # update history
                self.smooth_loss_history.append(smooth_loss)
                self.log_lr_history.append(np.log10(self.learning_rate))
                loss.backward()
                self.trainer.optimizer.step()

                # update learning rate
                self.learning_rate *= self.lr_mult
                self.trainer.set_learning_rate(self.learning_rate)

                # break if the loss gets too large
                if smooth_loss > self.explode_thresh * self.best_loss:
                    explode = True
                    print('[FIND_LR] loss hit explode threshold [%.3f x best (%f)]' %\
                          (self.explode_thresh, self.best_loss)
                    )
                    break

                # break if we've seen enough batches
                if self.max_batches > 0 and batch_idx >= self.max_batches:
                    break

            # need to also break out of the outer loop
            if explode is True:
                break

        # restore state
        print('[FIND_LR] : restoring trainer state')
        self.trainer.set_trainer_params(self.load_trainer_params())
        print('[FIND_LR] : restoring model state')
        self.trainer.model.set_net_state_dict(self.load_model_params())

        return self.get_lr_range()

    def acc(self, data_loader, batch_idx) -> None:
        """
        acc()
        Collect accuracy stats while finding learning rate
        """
        val_loss = 0.0
        correct = 0
        self.trainer.model.set_eval()
        with torch.no_grad():
            for n, (data, labels) in enumerate(data_loader):
                data = data.to(self.trainer.device)
                labels = labels.to(self.trainer.device)

                output = self.trainer.model.forward(data)
                loss = self.trainer.criterion(output, labels)
                val_loss += loss.item()

                # accuracy
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum().item()

        avg_val_loss = val_loss / len(data_loader)
        acc = correct / len(data_loader.dataset)
        self.acc_history.append(acc)
        if batch_idx % self.print_every == 0:
            print('[FIND_LR ACC]  : Avg. Test Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
                (avg_val_loss, correct, len(data_loader.dataset),
                100.0 * acc)
            )



# Rather than try to have the LFFinder promote itself to the correct class when
# load() is called, this function wraps the class instantiation and then
# returns  an object of the original type with the correct state parameters
def lr_finder_auto_load(filename:str) -> LRFinder:
    with open(filename, 'rb') as fp:
        state = pickle.load(fp)
    imp = importlib.import_module(state['module_path'])
    mod = getattr(imp, state['module_name'])

    lr_finder = mod(None)
    lr_finder.load(filename)

    return lr_finder
