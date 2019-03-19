"""
SUPERRES
Super resolution trainer

Stefan Wong 2019
"""

from lernomatic.train import trainer


class SRTrainer(trainer.Trainer):
    def __init__(self, model, **kwargs):

        super(SRTrainer, self).__init__(model, **kwargs)

    def __repr__(self):
        return 'SRTrainer'

    # TODO : checkpointing, history, etc


    def train_epoch(self):
        self.model.train()

        for n, (lr, hr, idx_scale) in enumerate(self.train_loader):
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.criterion(sr, hr)
            loss.backward()

            # TODO: gradient clip here

            self.optimizer.step()

            # display

            # save

            # scheduling
