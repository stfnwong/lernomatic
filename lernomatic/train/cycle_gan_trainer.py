"""
CYCLE_GAN_TRAINER
Trainer for Cycle GAN model

Stefan Wong 2019
"""

import time
from lernomatic.train import trainer

class CycleGANTrainer(trainer.Trainer):
    def __init(self, model, **kwargs):
        super(CycleGANTrainer, self).__init__(model, **kwargs)


    def train_epoch(self):
        self.model.train()

        for n, (data, labels) in enumerate(self.train_loader):
            pass
