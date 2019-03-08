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
#from pudb import set_trace; set_trace()


class MNISTTrainer(trainer.Trainer):
    def __init__(self, model=None, **kwargs):
        self.data_dir       = kwargs.pop('data_dir', 'data/')
        super(MNISTTrainer, self).__init__(model, **kwargs)

        # init the criterion for MNIST
        self.criterion = torch.nn.NLLLoss()

    def __repr__(self):
        return 'MNISTTrainer'

    def _init_optimizer(self):
        if self.model is not None:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr = self.learning_rate,
                momentum = self.momentum
            )
        else:
            self.optimizer = None       # for when we load checkpoints from disk

    def _init_dataloaders(self):
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
        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                self.data_dir,
                train = False,
                download = True,
                transform = dataset_transform
            ),
            batch_size = self.test_batch_size,
            shuffle = self.shuffle
        )

    def save_history(self, fname):
        history = dict()
        history['loss_history']   = self.loss_history
        history['loss_iter']      = self.loss_iter
        history['cur_epoch']      = self.cur_epoch
        history['iter_per_epoch'] = self.iter_per_epoch
        if self.test_loss_history is not None:
            history['test_loss_history'] = self.test_loss_history

        torch.save(history, fname)

    def load_history(self, fname):
        history = torch.load(fname)
        self.loss_history   = history['loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch      = history['cur_epoch']
        self.iter_per_epoch = history['iter_per_epoch']
        if 'test_loss_history' in history:
            self.test_loss_history = history['test_loss_history']

    def save_checkpoint(self, fname):
        checkpoint = dict()
        checkpoint['model'] = self.model.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['trainer'] = self.get_trainer_params()
        torch.save(checkpoint, fname)

    def load_checkpoint(self, fname):
        checkpoint = torch.load(fname)
        self.set_trainer_params(checkpoint['trainer'])
        self.model = mnist.MNISTNet()
        self.model.load_state_dict(checkpoint['model'])
        self._init_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._init_device()
        self._send_to_device()
