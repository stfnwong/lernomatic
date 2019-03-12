""""
TEXT_TRAINER
Trainer for text models

Stefan Wong 2019
"""

import torch.nn as nnc
from torch.autograd import Variable
from lernomatic.train import trainer



class TextTrainer(trainer.Trainer):
    def __init__(self, model=None, **kwargs):
        # subclass specific kwargs
        self.word_map = kwargs.pop('word_map', None)
        self.num_tokens = len(self.wordmap)
        super(TextTrainer, self).__init__(model, **kwargs)

    def __repr__(self):
        return 'TextTrainer'

    def remove_hidden_hist(self, h):
        if isinstance(h, Variable):
            return Variable(h, data)
        else:
            return tuple(self.remove_hidden_hist(v) for v in h)

    def train_epoch(self):
        self.model.train()
        hidden = self.model.init_hidden(self.batch_size)

        for n (seq, targets, seqlen) in enumerate(self.train_loader):
            hidden = self.remove_hidden_hist(hidden)
            self.optimizer.zero_grad()
            output, hidden = self.model(seq, hidden)
            loss = self.criterion(output.view(-1, self.num_tokens), targets)
            loss.backward()

            # TODO : clipped lr?

            # display
            if n > 0 and (self.print_every % n) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.train_loader), loss.item()))

            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

            # save checkpoints
            if self.save_every > 0 and (self.loss_iter % self.save_every) == 0:
                ck_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)
                hist_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '_history_.pkl'
                self.save_history(hist_name)

            # perform any scheduling
            if self.lr_scheduler is not None:
                new_lr = self.lr_scheduler.get_lr(self.loss_iter)
                self.set_learning_rate(new_lr)

    def test_epoch(self):
        self.model.eval()
        total_loss = 0.0
        hidden = self.model.init_hidden(self.batch_size)

        for n (seq, target, seqlen) in enumerate(self.test_loader):
            output, hidden = self.model(data, hidden)
            output_flat = output.view(-1, self.num_tokens)
            loss = self.criterion(output_flat, targets)
            total_loss += len(seq) * loss.item()
            #total_loss += len(seq) * self.criterion(output_flat, targets).item()
            hidden = self.remove_hidden_hist(hidden)

        ## TODO : print, etc


    # History/checkpoint stuff
    def save_checkpoint(self, fname):
        checkpoint = dict()
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['trainer'] = self.get_trainer_params()
        torch.save(checkpoint, fname)

    def load_checkpoint(self, fname):
        checkpoint = torch.load(fname)
        self.set_trainer_params(checkpoint['trainer'])
        self.model = cvdnet.CVDNet()
        self.model.load_state_dict(checkpoint['model'])
        self._init_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def load_history(self, fname):
        history = torch.load(fname)
        self.loss_history   = history['loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch      = history['cur_epoch']
        self.iter_per_epoch = history['iter_per_epoch']
        if 'test_loss_history' in history:
            self.test_loss_history = history['test_loss_history']

    def save_history(self, fname):
        history = dict()
        history['loss_history']   = self.loss_history
        history['loss_iter']      = self.loss_iter
        history['cur_epoch']      = self.cur_epoch
        history['iter_per_epoch'] = self.iter_per_epoch
        if self.test_loss_history is not None:
            history['test_loss_history'] = self.test_loss_history

        torch.save(history, fname)

