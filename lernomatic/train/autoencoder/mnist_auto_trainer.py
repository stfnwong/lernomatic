"""
MNIST_AUTO_TRAINER
Example trainer for Autoencoder with MNIST dataset

Stefan Wong 2019
"""

import numpy as np
import torch
import torchvision
# save images
from torchvision.utils import save_image
# lernomatic modules
from lernomatic.models import common
from lernomatic.models import mnist
from lernomatic.train import trainer


def to_img(X : torch.Tensor) -> torch.Tensor:
    X = X.view(X.size(0), 1, 28, 28)
    return X

class MNISTAutoTrainer(trainer.Trainer):
    """
    Trainer for Autoencoder model. Trying to work out what is the best way
    to design the API for this
    """
    def __init__(self, model: common.LernomaticModel, **kwargs) -> None:
        self.save_img_every = kwargs.pop('save_img_every', 10)
        self.save_img_dir   = kwargs.pop('save_img_dir', './figures/')
        self.data_dir       = kwargs.pop('data_dir', 'data/')
        super(MNISTAutoTrainer, self).__init__(model, **kwargs)
        # AutoTrainer specific keywords
        #self.val_loader     = kwargs.pop('val_loader', None)
        # Init history
        self.criterion = torch.nn.MSELoss()
        self._init_history()

    def __repr__(self) -> str:
        return 'AutoTrainer-%d' % self.train_loader.batch_size

    def __str__(self) -> str:
        s = []
        s.append('AutoTrainer [%s]\n' % str(self.criterion))
        s.append('\tnum epochs    : %d\n' % self.num_epochs)
        s.append('\tlearning rate : %e\n' % self.learning_rate)
        s.append('\tweight decay  : %e\n' % self.weight_decay)
        s.append('\tcurrent epoch : %d\n' % self.cur_epoch)
        s.append('\tdevice        : %s\n' % self.get_device_str())

        return ''.join(s)

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
            num_workers = self.num_workers,
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
            num_workers = self.num_workers,
            shuffle = self.shuffle
        )

        self.test_loader = None

    def _init_history(self):
        self.loss_history = np.zeros(len(self.train_loader) * self.num_epochs)
        self.loss_iter = 0

    def save_history(self, fname: str) -> None:
        history = dict()
        history['loss_history']   = self.loss_history
        history['loss_iter']      = self.loss_iter
        history['cur_epoch']      = self.cur_epoch
        history['iter_per_epoch'] = self.iter_per_epoch
        if self.test_loss_history is not None:
            history['test_loss_history'] = self.test_loss_history

        torch.save(history, fname)

    def load_history(self, fname: str) -> None:
        history = torch.load(fname)
        self.loss_history   = history['loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch      = history['cur_epoch']
        self.iter_per_epoch = history['iter_per_epoch']
        if 'test_loss_history' in history:
            self.test_loss_history = history['test_loss_history']

    def train_epoch(self) -> None:
        self.model.set_train()
        for n, data in enumerate(self.train_loader):
            img, _ = data
            img = img.view(img.size(0), -1)
            img = img.to(self.device)

            # forward pass
            output = self.model.forward(img)
            loss = self.criterion(output, img)
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print some stats
            if (n % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.train_loader), loss.item()))

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('loss/train', loss.item(), self.loss_iter)

            # Save the output images
            #if n % self.save_img_every == 0 and n > 0:
            #    x = to_img(img.cpu().data)
            #    x_hat = to_img(output.cpu().data)
            #    save_image(x, '%sx_%d.png' % (self.save_img_dir, self.cur_epoch))
            #    save_image(x_hat, '%s/xhat_%d.png' % (self.save_img_dir, self.cur_epoch))

            # Record loss
            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

            if (self.tb_writer is not None) and (n % self.save_img_every == 0):
                x = to_img(img.cpu().data)
                x_hat = to_img(output.cpu().data)
                self.tb_writer.add_image('x', x, self.cur_epoch)
                self.tb_writer.add_image('x_hat', x_hat, self.cur_epoch)
                #save_image(x, '%sx_%d.png' % (self.save_img_dir, self.cur_epoch))
                #save_image(x_hat, '%s/xhat_%d.png' % (self.save_img_dir, self.cur_epoch))

    def train(self) -> None:
        if self.train_loader is None:
            raise ValueError('Internal dataloader not set')

        self._send_to_device()
        print('======== Starting training loop ========')
        for n in range(self.num_epochs):
            self.train_epoch()
            self.cur_epoch += 1
            # save checkpoints
            if n % self.save_every == 0 and n > 0:
                model_file = './checkpoint/auto_epoch_%d.pth' % int(n)
                print('Saving model to file %s (epoch %d)...' % (model_file, n))
                self.save_checkpoint(model_file)
