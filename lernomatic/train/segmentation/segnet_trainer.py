"""
SEGNET_TRAINER
Train the SegNet model

Stefan Wong 2019
"""


import torch
from lernomatic.models import common
from lernomatic.models.segmentation import segnet
from lernomatic.train import trainer



class SegnetTrainer(trainer.Trainer):
    def __init__(self,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.encoder = encoder
        self.decoder = decoder
        super(SegnetTraner, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'SegnetTrainer'

    def _init_optimizer(self) -> None:
        if self.encoder is None:
            self.enc_optim = None
        else:
            if hasattr(torch.optim, self.optim_function):
                self.enc_optim = getattr(torch.optim, self.optim_function)(
                    self.encoder.get_model_parameters(),
                    lr = self.learning_rate,
                    weight_decay = self.weight_decay,
                )
            else:
                raise ValueError('Cannot find optim function %s' % str(self.optim_function))

        if self.decoder is None:
            self.dec_optim = None
        else:
            if hasattr(torch.optim, self.optim_function):
                self.dec_optim = getattr(torch.optim, self.optim_function)(
                    self.encoder.get_model_parameters(),
                    lr = self.learning_rate,
                    weight_decay = self.weight_decay,
                )
            else:
                raise ValueError('Cannot find optim function %s' % str(self.optim_function))


    def train_epoch(self) -> None:
        for batch_idx, (data, label) in enumerate(self.train_loader):
            data = data.to(self.device)
            label = label.to(self.device)

