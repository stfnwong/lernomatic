"""
CVD_TRAINER
Trainer for Cats-vs-Dogs model

Stefan Wong 2018
"""

import torch
from lernomatic.train import trainer
from lernomatic.models import cvdnet
from lernomatic.models import common

# debug
#


class CVDTrainer(trainer.Trainer):
    def __init__(self, model:common.LernomaticModel, **kwargs) -> None:
        super(CVDTrainer, self).__init__(model, **kwargs)

    def __repr__(self) -> str:
        return 'CVDTrainer'

    def save_history(self, fname: str) -> None:
        history = dict()
        history['loss_history']   = self.loss_history
        history['loss_iter']      = self.loss_iter
        history['cur_epoch']      = self.cur_epoch
        history['iter_per_epoch'] = self.iter_per_epoch
        if self.test_loss_history is not None:
            history['test_loss_history'] = self.test_loss_history

        torch.save(history, fname)

    def load_history(self, fname: str) -> None:
        history = torch.load(fname)
        self.loss_history   = history['loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch      = history['cur_epoch']
        self.iter_per_epoch = history['iter_per_epoch']
        if 'test_loss_history' in history:
            self.test_loss_history = history['test_loss_history']

    def test_epoch(self) -> None:
        self.model.set_eval()
        test_loss = 0.0
        correct = 0

        for n, (data, labels) in enumerate(self.test_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            output = self.model.forward(data)
            loss = self.criterion(output, labels)
            test_loss += loss.item()

            # accuracy
            _, pred = torch.max(output, 1)
            correct += torch.sum(pred == labels.data).item()

            if (n % self.print_every) == 0:
                print('[TEST]  :   Epoch       iteration         Test Loss    Correct    Total ')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f    %d     %d' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.test_loader), loss.item(),
                       correct, len(self.test_loader.dataset))
                )

            self.test_loss_history[self.test_loss_iter] = loss.item()
            self.test_loss_iter += 1

        avg_test_loss = test_loss / len(self.test_loader)
        acc = correct / len(self.test_loader.dataset)
        self.acc_history[self.acc_iter] = acc
        self.acc_iter += 1
        print('[TEST]  : Avg. Test Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
              (avg_test_loss, correct, len(self.test_loader.dataset),
               100.0 * acc)
        )

        # save the best weights
        if acc > self.best_acc:
            self.best_acc = acc
            if self.save_every > 0:
                ck_name = self.checkpoint_dir + '/' + 'best_' +  self.checkpoint_name
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)
                hist_name = self.checkpoint_dir + '/' + 'best_' + self.checkpoint_name + '_history.pkl'
                if self.verbose:
                    print('\t Saving history to file [%s] ' % str(hist_name))
                self.save_history(hist_name)
