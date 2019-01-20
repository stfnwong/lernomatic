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
from pudb import set_trace; set_trace()

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

    def train_epoch(self):
        """
        TRAIN_EPOCH
        Perform training on the model for a single epoch of the dataset
        """
        self.model.train()
        # training loop
        for n, (data, target) in enumerate(self.train_loader):
            # move data
            data = data.to(self.device)
            target = target.to(self.device)

            # optimization
            self.optimizer.zero_grad()
            output = self.model(data)
            loss   = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if (n % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.train_loader), loss.item()))

            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

            # save checkpoints
            if self.save_every > 0 and (self.loss_iter % self.save_every) == 0:
                ck_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)
                hist_name = self.checkpoint_dir + '/' + self.checkpoint_name +\
                    '_iter_' + str(self.loss_iter) + '_epoch_' + str(self.cur_epoch) + '_history_.pkl'
                self.save_history(hist_name)


    def test_epoch(self):
        """
        TEST_EPOCH
        Perform testing on one epoch of the test data
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0.0

        with torch.no_grad():
            for n, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                if self.verbose:
                    print('[VAL]   : element [%d / %d]' % (n+1, len(self.test_loader)), end='\r')
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

        if self.verbose:
            print('\n ..done')

        test_loss /= len(self.test_loader)
        self.test_loss_history[self.cur_epoch] = correct / len(self.test_loader.dataset)
        # show output
        print('[VAL]   : Avg. Test Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
              (test_loss, correct, len(self.test_loader.dataset),
               100.0 * correct / len(self.test_loader.dataset))
        )

    def train(self):
        for n in range(self.num_epochs):
            self.train_epoch()

            if self.test_loader is not None:
                self.test_epoch()
            self.cur_epoch += 1

