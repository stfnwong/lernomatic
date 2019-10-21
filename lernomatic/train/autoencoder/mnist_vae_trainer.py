"""
MNIST VAE TRAINER
"""
import torch
import torchvision
from torch.nn import functional as F
from lernomatic.train import trainer
from lernomatic.models import common

# debug
#from pudb import set_trace; set_trace()

class MNISTVAETrainer(trainer.Trainer):
    """
    VAETRAINER
    Trainer adapted for Variational Autoencoders
    """
    def __init__(self, model:common.LernomaticModel, **kwargs) -> None:
        self.data_dir = kwargs.pop('data_dir', 'data/')
        super(MNISTVAETrainer, self).__init__(model, **kwargs)
        self.criterion = torch.nn.MSELoss()

        self._init_history()

    def __repr__(self) -> str:
        return 'VAETrainer-%d' % self.train_loader.batch_size

    def __str__(self) -> str:
        s = []
        s.append('VAETrainer [%s]\n' % str(self.criterion))
        s.append('\tnum epochs    : %d\n' % self.num_epochs)
        s.append('\tlearning rate : %e\n' % self.learning_rate)
        s.append('\tweight decay  : %e\n' % self.weight_decay)
        s.append('\tcurrent epoch : %d\n' % self.cur_epoch)

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
            batch_size  = self.val_batch_size,
            num_workers = self.num_workers,
            shuffle     = self.shuffle
        )

        self.test_loader = None

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

    def vae_loss_function(self, recon_x, x, mu, logvar) -> None:
        # compute loss between real X and reconstructed X
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.model.get_input_size()))
        # Compute KLD and add to loss function
        # more info in https://arxiv.org/abs/1312.6114 (appendix B)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # normalize by same number of elements as in reconstruction
        KLD /= self.train_loader.batch_size * self.model.get_input_size()

        return BCE+KLD

    def train_epoch(self) -> None:
        self.model.set_train()
        train_loss = 0.0
        # for MNIST len(train_loader.dataset) is 60000
        # each element of the data is a tensor of shape
        # (BATCH_SIZE, 128, 1, 28]
        for batch_idx, (data, _) in enumerate(self.train_loader):
            #data = Variable(data).to(self.torch_device)
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # ---- forward pass ---- #
            recon_batch, mu, logvar = self.model.forward(data)
            # calculate scalar loss
            loss = self.vae_loss_function(recon_batch, data, mu, logvar)

            # ---- backward pass ---- #
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            # print some stats
            if (batch_idx % self.print_every) == 0:
                print('[TRAIN] :   Epoch       iteration         Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_loader), loss.item()))

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('loss/train', loss.item(), self.loss_iter)

            self.loss_history[self.loss_iter] = loss.item()
            self.loss_iter += 1

            # Apply scheduling
            if self.lr_scheduler is not None:
                self.apply_lr_schedule()

        print('Epoch %d average loss : %.6f' % (self.cur_epoch, train_loss / len(self.train_loader.dataset)))

    def train(self) -> None:
        self._send_to_device()
        for n in range(self.num_epochs):
            self.train_epoch()

            if n % self.save_every == 0:
                model_file = './checkpoint/vae_epoch_%d.pth' % int(n)
                print('Saving model to file %s (epoch %d)...' % (model_file, n))
                self.save_checkpoint(model_file)
            self.cur_epoch += 1
