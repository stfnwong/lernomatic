"""
MNIST VAE TRAINER
"""
import torch
import torchvision
from torch.nn import functional as F
from lernomatic.train import trainer

# debug
from pudb import set_trace; set_trace()

class MNISTVAETrainer(trainer.Trainer):
    """
    VAETRAINER
    Trainer adapted for Variational Autoencoders
    """
    def __init__(self, model, **kwargs):
        self.data_dir = kwargs.pop('data_dir', 'data/')
        super(MNISTVAETrainer, self).__init__(model, **kwargs)

    def __repr__(self):
        return 'VAETrainer-%d' % self.train_loader.batch_size

    def __str__(self):
        s = []
        s.append('VAETrainer [%s]\n' % str(self.criterion))
        s.append('\tnum epochs    : %d\n' % self.num_epochs)
        s.append('\tlearning rate : %e\n' % self.learning_rate)
        s.append('\tweight decay  : %e\n' % self.weight_decay)
        s.append('\tcurrent epoch : %d\n' % self.cur_epoch)

        return ''.join(s)

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


    def save_checkpoint(self, fname):
        trainer_params = self.get_trainer_params()
        checkpoint_data = {
            'model' : self.model.state_dict(),
            'optim' : self.optimizer.state_dict(),
            'trainer_state' : trainer_params
        }
        torch.save(checkpoint_data, fname)

    def load_checkpoint(self, fname):
        checkpoint = torch.load(fname)
        self.set_trainer_params(checkpoint['trainer_params'])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(['optim'])

    def vae_loss_function(self, recon_x, x, mu, logvar):
        # compute loss between real X and reconstructed X
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.model.input_size))
        # Compute KLD and add to loss function
        # more info in https://arxiv.org/abs/1312.6114 (appendix B)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # normalize by same number of elements as in reconstruction
        KLD /= self.train_loader.batch_size * self.model.input_size

        return BCE+KLD

    def train_epoch(self):
        self.model.train()
        train_loss = 0.0
        # for MNIST len(train_loader.dataset) is 60000
        # each element of the data is a tensor of shape
        # (BATCH_SIZE, 128, 1, 28]
        for batch_idx, (data, _) in enumerate(self.train_loader):
            #data = Variable(data).to(self.torch_device)
            data = data.to(self.device)
            self.optimizer.zero_grad()

            # ---- forward pass ---- #
            recon_batch, mu, logvar = self.model(data)
            # calculate scalar loss
            loss = self.vae_loss_function(recon_batch, data, mu, logvar)

            # ---- backward pass ---- #
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            # log information
            if (batch_idx % self.print_every) == 0:
                print('Epoch <%d> [%d/%d] (%.0f %%)\t Loss : %.6f' %\
                    (self.epoch, batch_idx * len(data), len(self.train_loader.dataset),
                     100.0 * batch_idx / len(self.train_loader),
                    loss.item() / len(data))
                    )

        print('Epoch %d average loss : %.6f' % (self.epoch, train_loss / len(self.train_loader.dataset)))

    def train(self):
        self._send_to_device()
        self.epoch = 0;
        for n in range(self.num_epochs):
            self.train_epoch()

            if n % self.save_every == 0:
                model_file = './checkpoint/vae_epoch_%d.pth' % int(n)
                print('Saving model to file %s (epoch %d)...' % (model_file, n))
                torch.save(self.model.state_dict(), model_file)
            self.epoch += 1

