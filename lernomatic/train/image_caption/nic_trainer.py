"""
NIC_TRAINER
Trainer for Neural Image Caption Model.
This is sort of an experiment to simplify the ImageCaptionTrainer class.

Stefan Wong 2019
"""

import importlib
from lernomatic.model import common
from lernomatic.train import trainer


class NICTrainer(trainer.Trainer):
    def __init__(self,
                 encoder :common.LernomaticModel=None,
                 decoder :common.LernomaticModel=None,
                 **kwargs) -> None:

        self.encoder = encoder
        self.decoder = decoder
        super(NICTrainer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'NICTrainer'

    def __str__(self) -> str:
        s = []
        s.append('NICTrainer (%d epochs)\n' % self.num_epochs)
        params = self.get_trainer_params()
        s.append('Trainer parameters :\n')
        for k, v in params.items():
            s.append('\t [%s] : %s\n' % (str(k), str(v)))

        return ''.join(s)

    def _init_optimizer(self) -> None:
        if self.decoder is None:
            self.decoder_optim = None
        else:
            self.decoder_optim = torch.optim.Adam(
                params = filter(lambda p : p.requires_grad, self.decoder.get_model_parameters()),
                lr = self.dec_lr,
            )

        if self.encoder is None or (self.encoder.do_fine_tune() is False):
            self.encoder_optim = None
        else:
            self.encoder_optim = torch.optim.Adam(
                params = filter(lambda p : p.requires_grad, self.encoder.get_model_parameters()),
                lr = self.enc_lr,
            )

    def _send_to_device(self) -> None:
        if self.encoder is not None:
            self.encoder.send_to(self.device)
        if self.decoder is not None:
            self.decoder.send_to(self.device)

    def train_epoch(self) -> None:
        self.encoder.set_train()
        self.decoder.set_train()

        for batch_idx, (imgs, caps, caplens) in enumerate(self.train_loader):
            imgs = imgs.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)

            # encoder images
            imgs = self.encoder.forward(imgs)

            # TODO : pack_padded_sequence()



    def save_history(self, fname:str) -> None:
        history = dict()
        history['loss_history']   = self.loss_history
        history['loss_iter']      = self.loss_iter
        history['cur_epoch']      = self.cur_epoch
        history['iter_per_epoch'] = self.iter_per_epoch
        if self.val_loss_history is not None:
            history['val_loss_history'] = self.val_loss_history
            history['val_loss_iter']    = self.val_loss_iter
        if self.acc_history is not None:
            history['acc_history'] = self.acc_history
            history['acc_iter']    = self.acc_iter

        torch.save(history, fname)

    def load_history(self, fname:str) -> None:
        history = torch.load(fname)
        self.loss_history   = history['loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch      = history['cur_epoch']
        self.iter_per_epoch = history['iter_per_epoch']
        if 'val_loss_history' in history:
            self.val_loss_history  = history['val_loss_history']
        if 'acc_history' in history:
            self.acc_history       = history['acc_history']
            self.acc_iter          = history['acc_iter']
        else:
            self.acc_iter = 0

    def save_checkpoint(self, fname: str) -> None:
        trainer_params = self.get_trainer_params()
        checkpoint_data = {
            # networks
            'encoder' : self.encoder.get_params(),
            'decoder' : self.decoder.get_params(),
            # solvers
            'encoder_optim' : self.encoder_optim.state_dict() if self.encoder_optim is not None else None,
            'decoder_optim' : self.decoder_optim.state_dict() if self.decoder_optim is not None else None
            # object params
            'trainer_state' : trainer_params,
        }
        torch.save(checkpoint_data, fname)

    def load_checkpoint(self, fname: str) -> None:
        checkpoint = torch.load(fname)
        self.set_trainer_params(checkpoint['trainer_state'])

        # Create the model objects and load parameters
        enc_model_path = checkpoint['encoder']['model_import_path']
        imp = importlib.import_module(enc_model_path)
        mod = getattr(imp, checkpoint['encoder']['model_name'])
        self.encoder = mod()
        self.encoder.set_params(checkpoint['encoder'])

        dec_model_path = checkpoint['decoder']['model_import_path']
        imp = importlib.import_module(dec_model_path)
        mod = getattr(imp, checkpoint['decoder']['model_name'])
        self.decoder = mod()
        self.decoder.set_params(checkpoint['decoder'])

        # load weights from checkpoint
        self._init_optimizer()
        self.decoder_optim.load_state_dict(checkpoint['decoder_optim'])
        self.encoder_optim.load_state_dict(checkpoint['encoder_optim'])

        # transfer decoder optimizer
        for state in self.decoder_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        # transfer encoder optimizer
        for state in self.encoder_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self._send_to_device()

# TODO : NIC Trainer with no encoder?
