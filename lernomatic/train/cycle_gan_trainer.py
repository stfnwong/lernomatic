"""
CYCLE_GAN_TRAINER
Trainer for Cycle GAN model

Stefan Wong 2019
"""

import time
from lernomatic.train import trainer

# debug
from pudb import set_trace; set_trace()


class CycleGANTrainer(trainer.Trainer):
    """
    CycleGANTrainer
    Trainer object for CycleGAN experiments

    """
    def __init(self, model, **kwargs):
        super(CycleGANTrainer, self).__init__(model, **kwargs)


    def train_epoch(self):
        self.model.train()

        for n, (data, labels) in enumerate(self.train_loader):
            pass

    #def test_epoch(self):
    #    self.model.eval()

    #    for n (data, labels) in
