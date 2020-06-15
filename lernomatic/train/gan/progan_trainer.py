"""
PROGAN_TRAINER
Trainer for a ProGAN model
Much of this is cobbled together from other places.

Stefan Wong 2020
"""
from lernomatic.train import trainer
from lernomatic.models import common

class ProGANTrainer(trainer.Trainer):
    def __init__(self, G=None:common.LernomaticModel, D=None:common.LernomaticModel, **kwargs) -> None:
        # models
        self.discriminator = D;
        self.generator = G
        # keyword args
        self.beta1         :float = kwargs.pop('beta1', 0.5)
        self.real_label    :int   = kwargs.pop('real_label', 1)
        self.fake_label    :int   = kwargs.pop('fake_label', 0)
        self.resl          :int   = kwargs.pop('resl', 2)           # resolution (2^2 = 4)
        self.phase         :str   = kwargs.pop('phase', 'init')

        super(ProGANTrainer, self).__init__(None, **kwargs)

    def train_epoch(self) -> None:
        for n, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            # Update D NETWORK
            # Maximize log(D(x)) + log(1 - D(G(z)))
            self.discriminator.zero_grad()


            self.generator.zero_grad()
