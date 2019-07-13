"""
CVD_TRAINER
Trainer for Cats-vs-Dogs model

Stefan Wong 2018
"""

import torch
from lernomatic.train import trainer
from lernomatic.models import common
from lernomatic.models.cvd import cvdnet

# debug
#from pudb import set_trace; set_trace()


class CVDTrainer(trainer.Trainer):
    def __init__(self, model:common.LernomaticModel, **kwargs) -> None:
        super(CVDTrainer, self).__init__(model, **kwargs)

    def __repr__(self) -> str:
        return 'CVDTrainer'

    def save_history(self, fname:str) -> None:
        history = dict()
        history['loss_history']   = self.loss_history
        history['loss_iter']      = self.loss_iter
        history['cur_epoch']      = self.cur_epoch
        history['iter_per_epoch'] = self.iter_per_epoch
        if self.val_loss_history is not None:
            history['val_loss_history'] = self.val_loss_history

        torch.save(history, fname)

    def load_history(self, fname:str) -> None:
        history = torch.load(fname)
        self.loss_history   = history['loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch      = history['cur_epoch']
        self.iter_per_epoch = history['iter_per_epoch']
        if 'val_loss_history' in history:
            self.val_loss_history = history['val_loss_history']

    def val_epoch(self) -> None:
        """
        VAL_EPOCH
        Run a single epoch of validation
        """
        self.model.set_eval()
        val_loss = 0.0
        correct  = 0

        for n, (data, labels) in enumerate(self.val_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            output = self.model(data)
            loss = self.criterion(output, labels)
            val_loss += loss.item()

            # accuracy
            _, pred = torch.max(output, 1)
            correct += torch.sum(pred == labels.data).item()

            if (n % self.print_every) == 0:
                print('[VAL ]  :   Epoch       iteration         Val Loss     Correct    Total ')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f    %d     %d' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.val_loader), loss.item(),
                       correct, len(self.val_loader.dataset))
                )

            self.val_loss_history[self.val_loss_iter] = loss.item()
            self.val_loss_iter += 1

        avg_val_loss = val_loss / len(self.val_loader)
        acc = correct / len(self.val_loader.dataset)
        self.acc_history[self.acc_iter] = acc
        self.acc_iter += 1
        print('[VAL ]  : Avg. Val Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
              (avg_val_loss, correct, len(self.val_loader.dataset), 100.0 * acc)
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
