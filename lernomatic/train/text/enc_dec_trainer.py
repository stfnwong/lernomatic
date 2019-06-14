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

# debug
#from pudb import set_trace; set_trace()


# custom batch object for this trainer
class RandBatch(object):
    def __init__(self, src:tuple, trg:tuple, pad_index:int=0) -> None:
        #src              = src[0]
        #src_lengths      = src[1]
        src, src_lengths = src

        self.src         = src
        self.src_lengths = src_lengths
        self.src_mask    = (src != pad_index).unsqueeze(-2)
        self.nseqs       = src.size(0)

        self.trg         = None
        self.trg_y       = None
        self.trg_mask    = None
        self.trg_legnths = None
        self.num_tokens  = None

        if trg is not None:
            trg, trg_lengths = trg
            self.trg = trg[:, :-1]
            self.trg_lengths = trg_lengths
            self.trg_y = trg[:, 1:]
            self.trg_mask = (self.trg_y != pad_index)
            self.ntokens = (self.trg_y != pad_index).data.sum().item()

    def __repr__(self) -> str:
        return 'RandBatch'

    def __str__(self) -> str:
        s = []
        s.append('RandBatch \n')
        s.append('src shape : %s\n' % str(self.src.shape))
        s.append('src lengths : %s\n' % str(self.src_lengths))

        if self.trg is not None:
            s.append('trg shape : %s\n' % str(self.trg.shape))
            s.append('trg lengths : %s\n' % str(self.trg_lengths))
            s.append('trg_y shape : %s\n' % str(self.trg_y.shape))

        return ''.join(s)

    #def __len__(self) -> int:
    #    return self.data.size(0)

    def to(self, device:torch.device) -> None:

        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)
        if self.trg is not None:
            self.trg = self.trg.to(device)
            self.trg_y = self.trg_y.to(device)
            self.trg_mask = self.trg_mask.to(device)


# generator for synthetic translation task
def data_gen(num_words:int = 11,
             batch_size:int = 16,
             num_batches:int = 100,
             length:int = 10,
             pad_index:int = 0,
             sos_index:int = 1,
             device=torch.device('cpu')) -> RandBatch:
    for i in range(num_batches):
        data = torch.from_numpy(
            np.random.randint(1, num_words, size=(batch_size, length))
        )
        data[:, 0] = sos_index
        data = data.to(device)
        src = data[:, 1:]
        trg = data
        src_lengths = [length-1] * batch_size
        trg_lengths = [length] * batch_size

        # note: data needs to have shape (batch_size, time, dim)

        yield RandBatch((src, src_lengths), (trg, trg_lengths), pad_index=pad_index)


# A sort of hack dataloader
class RandWordBatchGenerator(object):
    def __init__(self, **kwargs) -> None:
        self.num_words   :int = kwargs.pop('num_words', 11)
        self.batch_size  :int = kwargs.pop('batch_size', 16)
        self.embed_dim   :int = kwargs.pop('embed_dim', 64)
        self.length      :int = kwargs.pop('length', 10)
        self.pad_index   :int = kwargs.pop('pad_index', 0)
        self.sos_index   :int = kwargs.pop('sos_index', 1)
        self.num_batches :int = kwargs.pop('num_batches', 64)

    def __len__(self) -> int:
        return self.num_batches

    def __repr__(self) -> str:
        return 'RandWordBatchGenerator'

    def __getitem__(self, idx:int) -> RandBatch:
        if idx > len(self):
            raise StopIteration

        data = torch.from_numpy(
            np.random.randint(1, self.num_words, size=(self.batch_size, self.length))
            #np.random.randint(1, self.num_words, size=(self.batch_size, self.length, self.num_words))
            #np.random.randint(1, self.num_words, size=(self.batch_size, self.length, self.embed_dim))
        )
        data[:, 0] = self.sos_index
        src = data[:, 1:]
        trg = data
        src_lengths = [self.length-1] * self.batch_size
        trg_lengths = [self.length] * self.batch_size

        return RandBatch((src, src_lengths), (trg, trg_lengths), pad_index=self.pad_index)



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




# throwaway test for RandBatch and RandBatchGenerator
if __name__ == '__main__':


    data = data_gen()
