"""
SEQ2SEQ TRAINER

Stefan Wong 2019
"""

import importlib
import torch
from lernomatic.train import trainer
from lernomatic.models import common




class Seq2SeqTrainer(trainer.Trainer):
    def __init__(self,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.encoder = encoder
        self.decoder = decoder
        super(Seq2SeqTrainer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'Seq2SeqTrainer'

    def _send_to_device(self) -> None:
        if self.encoder is not None:
            self.encoder.send_to(self.device)
        if self.decoder is not None:
            self.decoder.send_to(self.device)

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
        pass

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
