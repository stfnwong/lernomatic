"""
IMAGE_CAPTION_LR
Learning rate finders for image caption trainers

Stefan Wong 2019
"""

import copy
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from lernomatic.train import trainer
from lernomatic.param import learning_rate

# debug
#from pudb import set_trace; set_trace()


class CaptionLogFinder(learning_rate.LRFinder):
    def __init__(self, trainer: trainer.Trainer, **kwargs) -> None:
        super(CaptionLogFinder, self).__init__(trainer, **kwargs)

    def __repr__(self) -> str:
        return 'CaptionLogFinder'

    def __str__(self) -> str:
        s = []
        s.append('CaptionLogFinder. lr range [%f -> %f]\n' % (self.lr_min, self.lr_max))
        return ''.join(s)

    def save_decoder_params(self, params:dict) -> None:
        self.decoder_params = copy.deepcopy(params)

    def save_encoder_params(self, params:dict) -> None:
        self.encoder_params = copy.deepcopy(params)

    def load_decoder_params(self) -> dict:
        return self.decoder_params

    def load_encoder_params(self) -> dict:
        return self.encoder_params

    def find(self) -> tuple:
        self.check_loaders()
        # cache parameters for later
        #self.save_model_params(self.trainer.get_model_params())
        self.save_decoder_params(self.trainer.decoder.get_net_state_dict())
        if self.trainer.encoder is not None:
            self.save_encoder_params(self.trainer.encoder.get_net_state_dict())
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
            for batch_idx, (imgs, caps, caplens) in enumerate(self.trainer.train_loader):
                self.trainer.decoder.set_train()
                if self.trainer.encoder is not None:
                    self.trainer.encoder.set_train()

                imgs = imgs.to(self.trainer.device)
                caps = caps.to(self.trainer.device)
                caplens = caplens.to(self.trainer.device)

                # We may have two optimizers here
                self.trainer.decoder_optim.zero_grad()
                if self.trainer.encoder_optim is not None:
                    self.trainer.encoder_optim.zero_grad()

                # do forward pass
                enc_imgs = self.trainer.encoder.forward(imgs)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = self.trainer.decoder.forward(enc_imgs, caps, caplens)
                targets = caps_sorted[:, 1:]
                scores_packed  = pack_padded_sequence(scores,  decode_lengths, batch_first=True)
                targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)
                loss = self.trainer.criterion(scores_packed[0], targets_packed[0])
                # add attention regularization
                loss += self.trainer.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
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

                # Take a step
                loss.backward()
                self.trainer.decoder_optim.step()
                if self.trainer.encoder_optim is not None:
                    self.trainer.encoder_optim.step()

                # update learning rate
                self.learning_rate *= self.lr_mult
                self.trainer.set_learning_rate(self.learning_rate) # TODO : enc_lr and dec_lr?

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
        #self.trainer.model.set_net_state_dict(self.load_model_params())
        self.trainer.decoder.set_net_state_dict(self.load_decoder_params())
        if self.trainer.encoder is not None:
            self.trainer.encoder.set_net_state_dict(self.load_encoder_params())

        return self.get_lr_range()


    def acc(self, data_loader, batch_idx) -> None:
        """
        acc()
        Collect accuracy stats while finding learning rate
        """
        test_loss = 0.0
        correct = 0
        self.trainer.decoder.set_eval()
        if self.trainer.encoder is not None:
            self.trainer.encoder.set_eval()

        references = list()     # true captions for computing BLEU-4 score
        hypotheses = list()     # predicted captions
        correct = 0

        for n, (imgs, caps, caplens, allcaps) in enumerate(data_loader):
            imgs = imgs.to(self.trainer.device)
            caps = caps.to(self.trainer.device)
            caplens = caplens.to(self.trainer.device)
            allcaps = allcaps.to(self.trainer.device)

            # encode images
            enc_imgs = self.trainer.encoder.forward(imgs)
            # do forward pass
            scores, caps_sorted, decode_lengths, alphas, sort_ind = self.trainer.decoder.forward(enc_imgs, caps, caplens)
            targets        = caps_sorted[:, 1:]
            scores_copy    = scores.clone()
            scores_packed  = pack_padded_sequence(scores,  decode_lengths, batch_first=True)
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            loss = self.trainer.criterion(scores_packed[0], targets_packed[0])
            # add attention regularization
            loss += self.trainer.alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            #pred = output.data.max(1, keepdim=True)[1]

            # accuracy
            # references
            allcaps = allcaps[sort_ind]     # decoder sorts images, we re-order here to match
            for cap_idx in range(allcaps.shape[0]):
                img_caps = allcaps[cap_idx].tolist()
                # remove <start> and <pad> tokens
                img_captions = list(
                    map(lambda c: \
                        [w for w in c if w not in {self.trainer.word_map.word_map['<start>'], self.trainer.word_map.word_map['<pad>']}], img_caps)
                )

                references.append(img_captions)

            # hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for n, pred in enumerate(preds):
                temp_preds.append(preds[n][0 : decode_lengths[n]])
            preds = temp_preds
            hypotheses.extend(preds)

            if len(references) != len(hypotheses):
                raise ValueError('len(references) <%d> != len(hypotheses) <%d>' %\
                                 (len(references), len(hypotheses))
                )
            #correct += pred.eq(labels.data.view_as(pred)).sum().item()

        bleu4 = corpus_bleu(references, hypotheses)
        avg_test_loss = test_loss / len(data_loader)
        #acc = correct / len(data_loader.dataset)
        self.acc_history.append(bleu4)
        if batch_idx % self.print_every == 0:
            print('[FIND_LR ACC]  : Avg. Test Loss : %.4f, BLEU4 (%.4f)' %\
                (avg_test_loss, bleu4)
            )
