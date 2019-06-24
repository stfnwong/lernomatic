"""
SUPERRES
Super resolution trainer

Stefan Wong 2019
"""

from lernomatic.models import common
from lernomatic.train import trainer


class SRTrainer(trainer.Trainer):
    def __init__(self, model:common.LernomaticModel, **kwargs) -> None:
        super(SRTrainer, self).__init__(model, **kwargs)

    def __repr__(self) -> str:
        return 'SRTrainer'

    # TODO : checkpointing, history, etc


    def train_epoch(self) -> None:
        self.model.set_train()

        for n, (lr, hr, idx_scale) in enumerate(self.train_loader):
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            self.optimizer.zero_grad()
            sr = self.model.forward(lr, idx_scale)
            loss = self.criterion(sr, hr)
            loss.backward()

            # TODO: gradient clip here

            self.optimizer.step()

            # display

            # save

            # scheduling
