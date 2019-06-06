"""
SEQ2SEQ TRAINER

Stefan Wong 2019
"""

import importlib
import torch
from lernomatic.train import trainer
from lernomatic.models import common
from lernomatic.data.text import vocab


# debug
from pudb import set_trace; set_trace()


class Seq2SeqTrainer(trainer.Trainer):
    def __init__(self,
                 voc:vocab.Vocabulary,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.voc     = voc
        self.use_teacher_forcing:bool = kwargs.pop('use_teacher_forcing', True)
        super(Seq2SeqTrainer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'Seq2SeqTrainer'

    def _send_to_device(self) -> None:
        if self.encoder is not None:
            self.encoder.send_to(self.device)
        if self.decoder is not None:
            self.decoder.send_to(self.device)

    def _init_optimizer(self) -> None:
        if self.encoder is not None:
            if hasattr(torch.optim, self.optim_function):
                self.enc_optim = getattr(torch.optim, self.optim_function)(
                    self.encoder.get_model_parameters(),
                    lr = self.learning_rate,
                    weight_decay = self.weight_decay
                )
            else:
                self.enc_optim = None

        if self.decoder is not None:
            if hasattr(torch.optim, self.optim_function):
                self.dec_optim = getattr(torch.optim, self.optim_function)(
                    self.decoder.get_model_parameters(),
                    lr = self.learning_rate,
                    weight_decay = self.weight_decay
                )
            else:
                self.dec_optim = None

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

        for batch_idx, (query, qlen, response, rlen) in enumerate(self.train_loader):
            query = query.to(self.device)
            response = response.to(self.device)

            # sort data
            query = query.sort(dim=1, descending=True)
            response = response.sort(dim=1, descending=True)


            print('Query vectors as strings...')
            for q in query:
                print(vocab.vec2sentence(q, self.voc))

            print('Response vectors as strings...')
            for r in response:
                print(vocab.vec2sentence(r, self.voc))

            # create mask for response sequence
            #r_mask = response > torch.LongTensor([self.voc.get_pad()])



    def val_epoch(self) -> None:
        pass


    # TODO : history, etc

    # history

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
        self.decoder = mod()
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
        history['test_loss_history'] = self.test_loss_history
        history['test_loss_iter']    = self.test_loss_iter
        history['acc_history']       = self.acc_history
        history['acc_iter']          = self.acc_iter
        history['cur_epoch']         = self.cur_epoch
        history['iter_per_epoch']    = self.iter_per_epoch
        if self.test_loss_history is not None:
            history['test_loss_history'] = self.test_loss_history

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
