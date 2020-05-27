"""
MNIST_TRAINER
Example trainer for MNIST dataset

Stefan Wong 2019
"""

import torch
import torchvision
from lernomatic.train import trainer
from lernomatic.models import mnist

# debug
#


class MNISTTrainer(trainer.Trainer):
    def __init__(self, model=None, **kwargs) -> None:
        self.data_dir :str = kwargs.pop('data_dir', 'data/')
        super(MNISTTrainer, self).__init__(model, **kwargs)

        # init the criterion for MNIST
        self.criterion = torch.nn.NLLLoss()

    def __repr__(self) -> str:
        return 'MNISTTrainer'

    def _init_optimizer(self) -> None:
        if self.model is not None:
            self.optimizer = torch.optim.SGD(
                self.model.get_model_parameters(),
                lr = self.learning_rate,
                momentum = self.momentum
            )
        else:
            self.optimizer = None       # for when we load checkpoints from disk

    def _init_dataloaders(self) -> None:
        dataset_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize( (0.1307,), (0.3081,))
        ])

        # training data
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                self.data_dir,
                train = True,
                download = True,
                transform = dataset_transform
            ),
            batch_size = self.batch_size,
            shuffle = self.shuffle
        )
        # validation data
        self.val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                self.data_dir,
                train = False,
                download = True,
                transform = dataset_transform
            ),
            batch_size = self.val_batch_size,
            shuffle = self.shuffle
        )

        self.test_loader = None

    def train_epoch(self) -> None:
        super(MNISTTrainer, self).train_epoch()

        # If we have a tensorboard writer, then visualize some outputs here
        #if self.tb_writer is not None:
        #    data, label = next(iter(self.train_loader))
        #    self.tb_writer.add_graph(self.model.net, data)

    def save_history(self, fname: str) -> None:
        history = dict()
        history['loss_history']   = self.loss_history
        history['loss_iter']      = self.loss_iter
        history['cur_epoch']      = self.cur_epoch
        history['iter_per_epoch'] = self.iter_per_epoch
        if self.val_loss_history is not None:
            history['val_loss_history'] = self.val_loss_history

        torch.save(history, fname)

    def load_history(self, fname: str) -> None:
        history = torch.load(fname)
        self.loss_history   = history['loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch      = history['cur_epoch']
        self.iter_per_epoch = history['iter_per_epoch']
        if 'val_loss_history' in history:
            self.val_loss_history = history['val_loss_history']
