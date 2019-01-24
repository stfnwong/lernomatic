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
from pudb import set_trace; set_trace()


class ResnetTrainer(trainer.Trainer):
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

    def train_epoch(self):
        self.model.train()
        for n, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (n % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.train_loader), loss.item()))

            if (n % self.save_every) == 0 and n > 0:
                ck_name = self.checkpoint_dir + self.checkpoint_name + '_iter_' + str(self.loss_iter) +\
                    '_epoch_' + str(self.cur_epoch) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s]' % str(ck_name))
                self.save_checkpoint(ck_name)

            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

    def test_epoch(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0


    def train(self):

        for n in range(self.num_epochs):
            self.train_epoch()

            if self.test_loader is not None:
                self.test_epoch()

            self.cur_epoch += 1
