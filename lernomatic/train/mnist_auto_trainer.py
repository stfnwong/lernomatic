"""
MNIST_AUTO_TRAINER
Example trainer for Autoencoder with MNIST dataset

Stefan Wong 2019
"""

import numpy as np
import torch
import torchvision
from lernomatic.train import trainer

# debug
#from pudb import set_trace; set_trace()

class MNISTAutoTrainer(trainer.Trainer):
    """
    Trainer for Autoencoder model. Trying to work out what is the best way
    to design the API for this
    """
    def __init__(self, model, **kwargs):
        self.save_img_every = kwargs.pop('save_img_every', 10)
        self.data_dir       = kwargs.pop('data_dir', 'data/')
        super(MNISTAutoTrainer, self).__init__(model, **kwargs)
        # AutoTrainer specific keywords
        #self.val_loader     = kwargs.pop('val_loader', None)
        # Init history
        self._init_history()

    def __repr__(self):
        return 'AutoTrainer-%d' % self.train_loader.batch_size

    def __str__(self):
        s = []
        s.append('AutoTrainer [%s]\n' % str(self.criterion))
        s.append('\tnum epochs    : %d\n' % self.num_epochs)
        s.append('\tlearning rate : %e\n' % self.learning_rate)
        s.append('\tweight decay  : %e\n' % self.weight_decay)
        s.append('\tcurrent epoch : %d\n' % self.cur_epoch)
        s.append('\tdevice        : %s\n' % self.get_device_str())

        return ''.join(s)

    def __eq__(self, other):
        pass

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
            num_workers = self.num_workers,
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
            num_workers = self.num_workers,
            shuffle = self.shuffle
        )


    def _init_history(self):
        self.loss_history = np.zeros(len(self.train_loader) * self.num_epochs)
        self.loss_iter = 0

    def save_checkpoint(self, fname):
        trainer_params = self.get_trainer_params()
        checkpoint_data = {
            'model' : self.model.state_dict(),
            'optim' : self.optimizer.state_dict(),
            'trainer_state' : trainer_params
        }
        torch.save(checkpoint_data, fname)

    def load_checkpoint(self, fname):
        checkpoint_data = torch.load(fname)
        self.set_trainer_params(checkpoint_data['trainer_params'])
        self.model.load_state_dict(checkpoint_data['model'])
        self.optimizer.load_state_dict(['optim'])

    def train_epoch(self):
        self.model.train()
        for n, data in enumerate(self.train_loader):
            img, _ = data
            img = img.view(img.size(0), -1)
            img = img.to(self.device)

            # forward pass
            output = self.model(img)
            loss = self.criterion(output, img)
            #mse_loss = nn.MSELoss()(output, img)
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print some stats
            if n % self.print_every == 0:
                print('Epoch <%d> [%d/%d] \t (%.0f %%)\t Loss : %.6f ' %\
                    (self.epoch+1, n * len(data), len(self.train_loader.dataset),
                     100.0 * n / len(self.train_loader),
                    loss.item())
                )

            # Save the output images
            #if n % self.save_img_every == 0:
            #    x = to_img(img.cpu().data)
            #    x_hat = to_img(output.cpu().data)
            #    save_image(x, './mlp_img/x_%d.png' % self.epoch)
            #    save_image(x_hat, './mlp_img/xhat_%d.png' % self.epoch)

            # Record loss
            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

    def train(self):
        if self.train_loader is None:
            raise ValueError('Internal dataloader not set')

        self._send_to_device()
        self.epoch = 0
        print('======== Starting training loop ========')
        for n in range(self.num_epochs):
            self.train_epoch()
            self.epoch += 1
            # save checkpoints
            if n % self.save_every == 0 and n > 0:
                model_file = './checkpoint/auto_epoch_%d.pth' % int(n)
                print('Saving model to file %s (epoch %d)...' % (model_file, n))
                torch.save(self.model.state_dict(), model_file)


#class MNISTTrainer(trainer.Trainer):
#    def __init__(self, model=None, **kwargs):
#        self.data_dir       = kwargs.pop('data_dir', 'data/')
#        super(MNISTTrainer, self).__init__(model, **kwargs)
#
#        # init the criterion for MNIST
#        self.criterion = torch.nn.NLLLoss()
#
#    def __repr__(self):
#        return 'MNISTTrainer'
#
#    def _init_optimizer(self):
#        if self.model is not None:
#            self.optimizer = torch.optim.SGD(
#                self.model.parameters(),
#                lr = self.learning_rate,
#                momentum = self.momentum
#            )
#        else:
#            self.optimizer = None       # for when we load checkpoints from disk
#
#    def _init_dataloaders(self):
#        dataset_transform = torchvision.transforms.Compose([
#            torchvision.transforms.ToTensor(),
#            torchvision.transforms.Normalize( (0.1307,), (0.3081,))
#        ])
#
#        # training data
#        self.train_loader = torch.utils.data.DataLoader(
#            torchvision.datasets.MNIST(
#                self.data_dir,
#                train = True,
#                download = True,
#                transform = dataset_transform
#            ),
#            batch_size = self.batch_size,
#            shuffle = self.shuffle
#        )
#        # validation data
#        self.test_loader = torch.utils.data.DataLoader(
#            torchvision.datasets.MNIST(
#                self.data_dir,
#                train = False,
#                download = True,
#                transform = dataset_transform
#            ),
#            batch_size = self.test_batch_size,
#            shuffle = self.shuffle
#        )
#
#    def save_history(self, fname):
#        history = dict()
#        history['loss_history']   = self.loss_history
#        history['loss_iter']      = self.loss_iter
#        history['cur_epoch']      = self.cur_epoch
#        history['iter_per_epoch'] = self.iter_per_epoch
#        if self.test_loss_history is not None:
#            history['test_loss_history'] = self.test_loss_history
#
#        torch.save(history, fname)
#
#    def load_history(self, fname):
#        history = torch.load(fname)
#        self.loss_history   = history['loss_history']
#        self.loss_iter      = history['loss_iter']
#        self.cur_epoch      = history['cur_epoch']
#        self.iter_per_epoch = history['iter_per_epoch']
#        if 'test_loss_history' in history:
#            self.test_loss_history = history['test_loss_history']
#
#    def save_checkpoint(self, fname):
#        checkpoint = dict()
#        checkpoint['model'] = self.model.state_dict()
#        checkpoint['optimizer'] = self.optimizer.state_dict()
#        checkpoint['trainer'] = self.get_trainer_params()
#        torch.save(checkpoint, fname)
#
#    def load_checkpoint(self, fname):
#        checkpoint = torch.load(fname)
#        self.set_trainer_params(checkpoint['trainer'])
#        self.model = mnist.MNISTNet()
#        self.model.load_state_dict(checkpoint['model'])
#        self._init_optimizer()
#        self.optimizer.load_state_dict(checkpoint['optimizer'])
#
