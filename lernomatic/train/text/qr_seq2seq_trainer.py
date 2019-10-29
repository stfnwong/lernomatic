"""
SEQ2SEQ TRAINER

Stefan Wong 2019
"""

import importlib
import torch
import numpy as np
from lernomatic.train import trainer
from lernomatic.models import common
from lernomatic.data.text import vocab


# debug
#from pudb import set_trace; set_trace()


# TODO : maybe this can eventually derive from trainer.text.Seq2SeqTrainer ?
class QRSeq2SeqTrainer(trainer.Trainer):
    """
    QRSeq2SeqTrainer
    Seq2Seq Trainer for Query/Response models (eg: chatbots)


    ARGUMENTS:
        encoder: (LernomaticModel)
            Encoder object.

        decoder: (LernomaticModel)
            Decoder object

        voc: (vocab.Vocabulary)
            Vocabulary object

        tf_rate: (float)
            Teacher forcing rate. Setting this to 0.0 ensures that teacher forcing never occurs in
            any given batch. Setting to 1.0 ensures teacher forcing always occurs in any given batch.
            Setting to any value in between roughly corresponds to the chance that teacher forcing will
            be used for that batch. The chance of teacher forcing for a given batch is computed as

            use_tf = np.random.random() < self.tf_rate

    """
    def __init__(self,
                 voc:vocab.Vocabulary=None,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.voc     :vocab.Vocabulary = voc
        self.encoder :common.LernomaticModel = encoder
        self.decoder :common.LernomaticModel = decoder
        # get subclass specific keyword args
        self.tf_rate   :float = kwargs.pop('tf_rate', 0.0)
        self.grad_clip :float = kwargs.pop('grad_clip', 0.0)
        #self.use_teacher_forcing:bool = kwargs.pop('use_teacher_forcing', True)
        super(QRSeq2SeqTrainer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'QRSeq2SeqTrainer'

    def _send_to_device(self) -> None:
        if self.encoder is not None:
            self.encoder.send_to(self.device)
        if self.decoder is not None:
            self.decoder.send_to(self.device)

    def _init_optimizer(self) -> None:
        if self.encoder is not None:
            if hasattr(torch.optim, self.optim_function):
                self.encoder_optim = getattr(torch.optim, self.optim_function)(
                    self.encoder.get_model_parameters(),
                    lr = self.learning_rate,
                    weight_decay = self.weight_decay
                )
            else:
                self.encoder_optim = None

        if self.decoder is not None:
            if hasattr(torch.optim, self.optim_function):
                self.decoder_optim = getattr(torch.optim, self.optim_function)(
                    self.decoder.get_model_parameters(),
                    lr = self.learning_rate,
                    weight_decay = self.weight_decay
                )
            else:
                self.decoder_optim = None

    def maskNLLLoss(self,
                    inp:torch.Tensor,
                    target:torch.Tensor,
                    mask:torch.Tensor) -> tuple:
        n_total = mask.sum()
        cross_entropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = cross_entropy.masked_select(mask).mean()
        loss = loss.to(self.device)

        return (loss, n_total.item())

    def train_epoch(self) -> None:

        self.encoder.set_train()
        self.decoder.set_train()
        decoder_initial_state = [[self.voc.get_sos() for _ in range(self.batch_size)]]

        for batch_idx, (query, qlen, response, rlen) in enumerate(self.train_loader):
            loss     = 0        # this becomes a tensor later...?
            totals   = 0

            # sort tensors in descending order of length
            qlen, qlen_sort_idx = qlen.squeeze(1).sort(dim=0, descending=True)
            rlen, rlen_sort_idx = rlen.squeeze(1).sort(dim=0, descending=True)
            query    = query[qlen_sort_idx]
            response = response[rlen_sort_idx]

            query = query.transpose(0, 1)
            response = response.transpose(0, 1)
            resp_mask = torch.where(
                response > self.voc.get_pad(),
                torch.ByteTensor([1]),
                torch.ByteTensor([0]),
            ).bool()
            resp_mask = resp_mask.to(self.device)
            query    = query.to(self.device)
            response = response.to(self.device)

            # pass oriented data to models
            enc_output, enc_hidden = self.encoder.forward(query, qlen)
            dec_input  = torch.LongTensor(decoder_initial_state)
            dec_input  = dec_input.to(self.device)
            dec_hidden = enc_hidden[0 : self.decoder.get_num_layers()]

            # decide if we are using teacher forcing this iteration
            tf_chance = np.random.random()
            if tf_chance < self.tf_rate:
                tf_this_batch = True
            else:
                tf_this_batch = False

            if tf_this_batch:
                for t in range(rlen.max().item()):
                    dec_output, dec_hidden = self.decoder(
                        dec_input, dec_hidden, enc_output
                    )
                    # teacher force - next input is current target
                    dec_input = response[t].view(1, -1)
                    # accumulate loss
                    mask_loss, n_total = self.maskNLLLoss(
                        dec_output,
                        response[t],
                        resp_mask[t]
                    )
                    loss += mask_loss
                    totals += n_total
            else:
                for t in range(rlen.max().item()):
                    dec_output, dec_hidden = self.decoder.forward(
                        dec_input, dec_hidden, enc_output
                    )
                    # next input is decoders own output
                    _, topk = dec_output.topk(1)
                    dec_input = torch.LongTensor([[topk[k][0] for k in range(self.batch_size)]])
                    dec_input = dec_input.to(self.device)
                    # accumulate loss
                    mask_loss, n_total = self.maskNLLLoss(
                        dec_output,
                        response[t],
                        resp_mask[t]
                    )
                    loss += mask_loss
                    totals += n_total

            self.encoder_optim.zero_grad()
            self.decoder_optim.zero_grad()
            loss.backward()

            self.encoder_optim.step()
            self.decoder_optim.step()

            # save loss history
            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

            # display
            if (batch_idx > 0) and (batch_idx % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss          Forced  chnc/rate')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f     %s [%.3f / %.3f]' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_loader),
                       loss.item(), str(tf_this_batch), tf_chance, self.tf_rate))

    def val_epoch(self) -> None:
        """
        VAL_EPOCH
        Perform validation on one epoch of validation data
        TODO : Write seperate beam searchers for this which can be brought in as modules
        """
        pass

    # checkpoints
    def save_checkpoint(self, filename:str) -> None:
        checkpoint = {
            # networks
            'encoder' : self.encoder.get_params(),
            'decoder' : self.decoder.get_params(),
            # solvers
            'encoder_optim' : self.encoder_optim.state_dict(),
            'decoder_optim' : self.decoder_optim.state_dict(),
            # trainer
            'trainer_params' : self.get_trainer_params()
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename:str) -> None:
        checkpoint_data = torch.load(filename)
        self.set_trainer_params(checkpoint_data['trainer_params'])

        # load encoder
        enc_model_path = checkpoint_data['encoder']['model_import_path']
        imp = importlib.import_module(enc_model_path)
        mod = getattr(imp, checkpoint_data['encoder']['model_name'])
        self.encoder = mod()
        self.encoder.set_params(checkpoint_data['encoder'])

        # load decoder
        dec_model_path = checkpoint_data['decoder']['model_import_path']
        imp = importlib.import_module(dec_model_path)
        mod = getattr(imp, checkpoint_data['decoder']['model_name'])
        self.decoder = mod(1, 1)
        self.decoder.set_params(checkpoint_data['decoder'])

        # load encoder optimizer
        self._init_optimizer()
        if checkpoint_data['decoder_optim'] is not None:
            self.decoder_optim.load_state_dict(checkpoint_data['decoder_optim'])
        if checkpoint_data['encoder_optim'] is not None:
            self.encoder_optim.load_state_dict(checkpoint_data['encoder_optim'])

        # transfers optimizer state to current device
        if self.encoder_optim is not None:
            for state in self.encoder_optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        if self.decoder_optim is not None:
            for state in self.encoder_optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        self._init_device()
        self._init_dataloaders()
        self._send_to_device()

    def save_history(self, fname: str) -> None:
        history = dict()
        history['loss_history']      = self.loss_history
        history['loss_iter']         = self.loss_iter
        history['acc_history']       = self.acc_history
        history['cur_epoch']         = self.cur_epoch
        history['iter_per_epoch']    = self.iter_per_epoch
        if self.test_loss_history is not None:
            history['test_loss_history'] = self.test_loss_history
            history['test_loss_iter']    = self.test_loss_iter

        if self.val_loss_history is not None:
            history['val_loss_history'] = self.val_loss_history
            history['acc_iter']          = self.acc_iter

        torch.save(history, fname)

    def load_history(self, fname: str) -> None:
        history = torch.load(fname)
        self.loss_history      = history['loss_history']
        self.loss_iter         = history['loss_iter']
        self.test_loss_history = history['test_loss_history']
        self.test_loss_iter    = history['test_loss_iter']
        self.acc_history       = history['acc_history']
        self.acc_iter          = history['acc_iter']
        self.cur_epoch         = history['cur_epoch']
        self.iter_per_epoch    = history['iter_per_epoch']
        if 'test_loss_history' in history:
            self.test_loss_history = history['test_loss_history']
