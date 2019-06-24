"""
ENC_DEC_TRAINER
A Trainer for the encoder/decoder in enc_dec_atten.py

Stefan Wong 2019
"""

import torch
import torch.nn as nn
import numpy as np
from lernomatic.models import common
from lernomatic.train import trainer
from lernomatic.data.text import rand_word

# debug
#from pudb import set_trace; set_trace()




# TODO : refactor to use data_gen instead of typical dataloaders
class EncDecTrainer(trainer.Trainer):
    def __init__(self,
                encoder:common.LernomaticModel,
                decoder:common.LernomaticModel,
                generator:common.LernomaticModel,
                **kwargs) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

        # TODO : this will be handled by the vocab in future
        self.num_words :int = kwargs.pop('num_words', 11)
        self.embed_dim :int = kwargs.pop('embed_dim', 32)
        self.train_loader   = kwargs.pop('train_loader', None)
        self.val_loader     = kwargs.pop('val_loader', None)

        super(EncDecTrainer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'EncDecTrainer'

    def _send_to_device(self) -> None:
        if self.encoder is not None:
            self.encoder.send_to(self.device)
        if self.decoder is not None:
            self.decoder.send_to(self.device)

    def _init_optimizer(self) -> None:
        # TODO: add support for completely independant optimizers (that have,
        # for example, independant learning rates)
        if self.encoder is not None:
            self.enc_optim = torch.optim.Adam(
                self.encoder.get_model_parameters(),
                lr = self.learning_rate,
                weight_decay = 0.0      # TODO : superconverge this?
            )
        else:
            self.enc_optim = None

        if self.decoder is not None:
            self.dec_optim = torch.optim.Adam(
                self.decoder.get_model_parameters(),
                lr = self.learning_rate,
                weight_decay = 0.0      # TODO : superconverge this?
            )
        else:
            self.dec_optim = None

        self.criterion = nn.NLLLoss(reduction='sum', ignore_index=0)

    def _init_dataloaders(self) -> None:
        # Need to implement this such that we use the batch/data_gen objects
        # above rather than the usual pytorch dataloaders
        self.train_loader = RandWordBatchGenerator(
            batch_size = self.batch_size,
            num_batches = self.num_epochs,
            num_words = self.num_words,      # TODO : make settable from vocab
            embed_dim = self.embed_dim
        )

        # TODO : hack for copy task
        self.val_loader = self.train_loader

    def compute_loss(self,
                     X:torch.Tensor,
                     y:torch.Tensor,
                     norm:float,
                     do_opt:bool=True) -> torch.Tensor:

        X = self.generator(X)
        loss = self.criterion(
            X.contiguous().view(-1, X.size(-1)),
            y.contiguous().view(-1)
        )
        loss = loss / norm

        if do_opt:
            loss.backward()
            self.enc_opt.step()
            self.dec_opt.step()
            # zero grads after taking a step
            self.enc_optim.zero_grad()
            self.dec_optim.zero_grad()

        return loss.data.item() * norm

    def train_epoch(self) -> None:
        if self.encoder is not None:
            self.encoder.set_train()
        if self.decoder is not None:
            self.decoder.set_train()

        total_loss = 0
        total_tokens = 0
        for batch_idx, batch in enumerate(self.train_loader):
            #batch = batch.to(self.device)
            batch.to(self.device)

            print('[TRAIN] : batch %d' % batch_idx)
            print('[TRAIN] : src : %s' % str(batch.src))
            print('[TRAIN] : trg : %s' % str(batch.trg))


            enc_out, enc_final = self.encoder.forward(
                batch.src,
                batch.src_mask,
                batch.src_lengths
            )
            dec_states, dec_hidden, pre_out = self.decoder.forward(
                enc_out,
                enc_final,
                batch.src_mask,
                batch.trg,
                batch.trg_mask
            )

            loss = self.compute_loss(pre_output, batch.trg_y, batch.nseqs, do_opt=True)
            total_loss += loss
            total_tokens += batch.ntokens

            if (batch_idx > 0) and (batch_idx % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_loader), loss)
                )

            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

            # TODO : checkpoints,

            # TODO : scheduling


    def eval_epoch(self) -> None:
        if self.encoder is not None:
            self.encoder.set_eval()
        if self.decoder is not None:
            self.decoder.set_eval()

        total_loss = 0
        total_tokens = 0
        perplexities = []

        # TODO : in the inital test we will use the same loader for train and
        # val
        for batch_idx, batch in enumerate(self.train_loader):
            #batch = batch.to(self.device)
            batch.to(self.device)

            enc_out, enc_final = self.encoder.forward(
                batch.src,
                batch.src_mask,
                batch.src_lengths
            )
            dec_states, dec_hidden, pre_out = self.decoder.forward(
                enc_out,
                enc_final,
                batch.src_mask,
                batch.trg,
                batch.trg_mask
            )

            loss = self.compute_loss(pre_output, batch.trg_y, batch.nseqs, do_opt=False)
            total_loss += loss
            total_tokens += batch.ntokens

            # TODO: this can go in history, if relevant
            perplexities.append(np.exp(total_loss / float(total_tokens)))

            if (batch_idx % self.print_every) == 0:
                print('[TEST]  :   Epoch       iteration         Test Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.test_loader), loss)
                )

            self.val_loss_history[self.val_loss_iter] = loss.item()
            self.val_loss_i += 1

        # eval condition
        #return np.exp(total_loss / float(total_tokens))



