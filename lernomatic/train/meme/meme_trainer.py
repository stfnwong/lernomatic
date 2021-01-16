"""
MEME_TRAINER
Training harness for imgflip meme generator

"""

from lernomatic.train.trainer import Trainer


class MemeTrainer(Trainer):
    def __init__(self, model, **kwargs):
        self.model = model
        self.data_dir       = kwargs.pop('data_dir', 'data/')
        super(MemeTrainer, self).__init__(model, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()
