"""
COCO_TRAIN_EXPR
An alternative trainer (that may get scrapped)

"""

import torch

from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from lernomatic.train import trainer



class COCOCaptTrainer(trainer.Trainer):
    """
    This is an attempt at a simpler caption trainer
    """
    def __init__(self, model, **kwargs):
        super(COCOCaptTrainer, self).__init__(model, **kwargs)

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
                params = filter(lambda p : p.requires_grad, self.model.parameters()),
                lr = self.learning_rate,
            )

    def clip_gradient(self):
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-self.grad_clip, self.grad_clip)

    def train_epoch(self):
        self.model.set_train()

        for n, (imgs, caps, caplens) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)

            self.optimizer.zero_grad()
            output = self.model.forward(imgs, caps, caplens)
            target_caps = pack_padded_sequence(caps, caplens)[0]
            loss = self.criterion(output, target_caps)
            loss.backward()

            if self.grad_clip > 0.0:
                self.clip_gradient()

            self.optimizer.step()

            # Display
            if (n % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.train_loader), loss.item()))

            # do scheduling
            if self.lr_scheduler is not None:
                new_lr = self.lr_scheduler.get_lr(self.loss_iter)
                self.set_learning_rate(new_lr)

            # update history
            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1
            # save checkpoint
            if n > 0 and (self.loss_iter % self.save_every == 0):
                ck_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)

    def test_epoch(self):
        self.model.set_eval()
        # for now, ignore allcaps output
        for n, (imgs, caps, caplens, _) in enumerate(self.test_loader):
            imgs = imgs.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)

            # test output
            output = self.model.forward(imgs, caps, caplens)
            target_caps = pack_padded_sequence(caps, caplens)[0]
            test_loss = self.criterion(output, target_caps)

            # display
            if (n % self.print_every) == 0:
                print('[TEST]  :   Epoch       iteration         Test Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.test_loader), test_loss.item()))

            self.test_loss_history[self.test_loss_iter] = test_loss.item()
            self.test_loss_iter += 1

    # TODO : checkpointing?
