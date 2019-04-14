"""
LR_COMMON
Tools for finding optimal learning rate

Stefan Wong 2018
"""

import numpy as np
import copy
import torch

# debug
#from pudb import set_trace; set_trace()


class LRFinder(object):
    """
    LRFinder
    Finds optimal learning rates
    """
    def __init__(self, trainer, **kwargs) -> None:
        valid_select_methods = ('max_acc', 'min_loss', 'max_range', 'min_range', 'laplace', 'sobel')
        self.trainer          = trainer
        # learning params
        self.num_epochs       = kwargs.pop('num_epochs', 8)
        # lr params
        self.lr_mult          :float = kwargs.pop('lr_mult', 0)
        self.lr_min           :float = kwargs.pop('lr_min', 1e-6)
        self.lr_max           :float = kwargs.pop('lr_max', 10)
        self.explode_thresh   :float = kwargs.pop('explode_thresh', 4.0)      # fast.ai uses 4 * min_smoothed_loss
        self.beta             :float = kwargs.pop('beta', 0.999)
        self.gamma            :float = kwargs.pop('gamma', 0.999995)
        self.lr_min_factor    :float = kwargs.pop('lr_min_factor', 2.0)
        self.lr_max_scale     :float = kwargs.pop('lr_max_scale', 1.0)
        self.lr_select_method :str   = kwargs.pop('lr_select_method', 'max_acc')
        self.lr_trunc         :int   = kwargs.pop('lr_trunc', 10)
        # gradient params
        self.grad_thresh      :float = kwargs.pop('grad_thresh', 0.002)
        # other
        self.acc_test         :bool  = kwargs.pop('acc_test', True)
        self.print_every      :int   = kwargs.pop('print_every', 20)
        self.verbose          :bool  = kwargs.pop('verbose', False)

        # trainer and model params
        self.model_params = None
        self.trainer_params = None
        # loss params
        self.avg_loss = 0.0
        self.best_loss = 1e6
        self.best_loss_idx = 0
        # acc params
        self.best_acc = 0.0
        self.best_acc_idx = 0
        # learning rate params
        self.learning_rate = 0.0

        self._init_history()

    def __repr__(self) -> str:
        return 'LRFinder'

    def __str__(self) -> str:
        s = []
        s.append('%s (%.4f -> %.4f)\n' % (repr(self), self.lr_min, self.lr_max))
        s.append('Method [%s]\n' % str(self.lr_select_method))
        return ''.join(s)

    def _print_find(self, epoch: int, batch_idx: int, loss: float) -> None:
        print('[FIND_LR] :  Epoch    iteration         loss    best loss (smooth)  lr')
        print('            [%d/%d]     [%6d/%6d]    %.6f     %.6f     %.6f' %\
            (epoch, self.num_epochs, batch_idx, len(self.trainer.train_loader),
            loss, self.best_loss, self.learning_rate)
        )

    def _print_acc(self, avg_test_loss: float, correct: int, acc: float, dataset_size: int) -> None:
        print('[FIND_LR ACC]  : Avg. Test Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
              (avg_test_loss, correct, dataset_size, 100.0 * acc)
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

    def _laplace_loss(self) -> tuple:
        k = np.array([-1, 4, -1])
        Xs = np.convolve(np.asarray(self.acc_history), k)
        clip_point = Xs[self.lr_trunc : len(Xs) - self.lr_trunc].max() * 0.9
        Xc = Xs[Xs < clip_point] = 0
        idxs = np.argwhere(Xs > 0)
        lr_max = self.log_lr_history[idxs[0][0]]
        lr_min = self.log_lr_history[idxs[-2][0]]

        return (lr_min, lr_max)

    def _sobel_loss(self) -> tuple:
        k = np.array([-1, 0, 1])
        Xs = np.convolve(np.asarray(self.acc_history), k)
        clip_point = Xs[self.lr_trunc : len(Xs) - self.lr_trunc].max() * 0.9
        Xc = Xs[Xs < clip_point] = 0
        idxs = np.argwhere(Xs > 0)
        lr_max = self.log_lr_history[idxs[0][0]]
        lr_min = self.log_lr_history[idxs[-2][0]]

        return (lr_min, lr_max)

    def save_model_params(self, params: dict) -> None:
        self.model_params = copy.deepcopy(params)

    def save_trainer_params(self, params: dict) -> None:
        self.trainer_params = copy.deepcopy(params)

    def load_model_params(self) -> dict:
        return self.model_params

    def load_trainer_params(self) -> dict:
        return self.trainer_params

    def check_loaders(self) -> None:
        if self.trainer.train_loader is None:
            raise ValueError('No train_loader in trainer')
        if self.trainer.test_loader is None:
            raise ValueError('No test_loader in trainer')

    def get_lr_history(self) -> list:
        if len(self.log_lr_history) < 1:
            return None
        return self.log_lr_history

    def find(self) -> tuple:
        raise NotImplementedError('This method should be implemented in subclass')

    def get_lr_range(self) -> tuple:
        if self.lr_select_method == 'max_acc':
            lr_max = self.log_lr_history[self.best_acc_idx] * self.lr_max_scale
            lr_min = lr_max * self.lr_min_factor
        elif self.lr_select_method == 'min_loss':
            lr_max = self.log_lr_history[self.best_loss_idx] * self.lr_max_scale
            lr_min = lr_max * self.lr_min_factor
        elif self.lr_select_method == 'max_range':
            lr_max = self.log_lr_history[-1] * self.lr_max_scale
            lr_min = lr_max * self.lr_min_factor
        elif self.lr_select_method == 'min_range':
            raise NotImplementedError('TODO : min_range method')
        elif self.lr_select_method == 'laplace':
            lr_min, lr_max = self._laplace_loss()
        elif self.lr_select_method == 'sobel':
            lr_min, lr_max = self._sobel_loss()
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

    def __str__(self) -> str:
        s = []
        s.append('LogFinder. lr range [%f -> %f]\n' % (self.lr_min, self.lr_max))
        return ''.join(s)

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
                    if self.trainer.test_loader is not None:
                        self.acc(self.trainer.test_loader, batch_idx)
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

                if smooth_loss > self.explode_thresh * self.best_loss:
                    explode = True
                    print('[FIND_LR] loss hit explode threshold [%.3f x best (%f)]' %\
                          (self.explode_thresh, self.best_loss)
                    )
                    break

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
        test_loss = 0.0
        correct = 0
        self.trainer.model.set_eval()
        with torch.no_grad():
            for n, (data, labels) in enumerate(data_loader):
                data = data.to(self.trainer.device)
                labels = labels.to(self.trainer.device)

                output = self.trainer.model.forward(data)
                loss = self.trainer.criterion(output, labels)
                test_loss += loss.item()

                # accuracy
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.data.view_as(pred)).sum().item()

        avg_test_loss = test_loss / len(data_loader)
        acc = correct / len(data_loader.dataset)
        self.acc_history.append(acc)
        if batch_idx % self.print_every == 0:
            print('[FIND_LR ACC]  : Avg. Test Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
                (avg_test_loss, correct, len(data_loader.dataset),
                100.0 * acc)
            )
