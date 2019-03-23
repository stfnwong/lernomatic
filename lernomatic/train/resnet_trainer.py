"""
RESNET_TRAINER
Trainer for Resnet models

Stefan Wong 2019
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from lernomatic.train import trainer
from lernomatic.models import resnets

# debug
#from pudb import set_trace; set_trace()


class ResnetTrainer(trainer.Trainer):
    """
    RESNETTRAINER
    Trainer object for resnet experiments
    """
    def __init__(self, model, **kwargs):
        self.data_dir      = kwargs.pop('data_dir', 'data/')
        self.augment_data  = kwargs.pop('augment_data', False)
        self.train_dataset = kwargs.pop('train_dataset', None)
        self.test_dataset  = kwargs.pop('test_dataset', None)
        super(ResnetTrainer, self).__init__(model, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()

    def __repr__(self):
        return 'ResnetTrainer'

    def __str__(self):
        s = []
        s.append('ResnetTrainer \n')
        if self.train_loader is not None:
            s.append('Training set size :%d\n' % len(self.train_loader.dataset))
        else:
            s.append('Training set not loaded\n')
        if self.test_loader is not None:
            s.append('Test set size :%d\n' % len(self.train_loader.dataset))
        else:
            s.append('Test set not loaded\n')

        return ''.join(s)

    def _init_dataloaders(self):
        """
        _INIT_DATALOADERS
        Generate dataloaders
        """
        normalize = transforms.Normalize(
            mean = [x / 255.0 for x in [125.3, 123.0, 113.9]],
            std  = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        )

        if self.augment_data:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x : F.pad(x.unsqueeze(0), (4,4,4,4), mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        if self.train_dataset is None:
            self.train_dataset = torchvision.datasets.CIFAR10(
                self.data_dir,
                train=True,
                download=True,
                transform=train_transform
            )
        if self.test_dataset is None:
            self.test_dataset = torchvision.datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=True,
                transform=test_transform
            )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle=self.shuffle,
            num_workers = self.num_workers
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            num_workers = self.num_workers
        )

    def save_history(self, fname):
        history = dict()
        history['loss_history']      = self.loss_history
        history['loss_iter']         = self.loss_iter
        history['test_loss_history'] = self.test_loss_history
        history['test_loss_iter']    = self.test_loss_iter
        history['acc_history']       = self.acc_history
        history['acc_iter']          = self.acc_iter
        history['cur_epoch']         = self.cur_epoch
        history['iter_per_epoch']    = self.iter_per_epoch
        if self.test_loss_history is not None:
            history['test_loss_history'] = self.test_loss_history

        torch.save(history, fname)

    def load_history(self, fname):
        history = torch.load(fname)
        self.loss_history      = history['loss_history']
        self.loss_iter         = history['loss_iter']
        self.test_loss_history = history['test_loss_history']
        self.test_loss_iter    = history['test_loss_iter']
        self.acc_history       = history['acc_history']
        self.acc_iter          = history['acc_iter']
        self.cur_epoch         = history['cur_epoch']
        self.iter_per_epoch    = history['iter_per_epoch']
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
        self.model = resnets.WideResnet(28, 10, 1)   # state dict overwrites values?
        self.model.load_state_dict(checkpoint['model'])
        self._init_optimizer()
        self.optimizer.load_state_dict(checkpoint['optimizer'])
