"""
ADVERSARIAL_SEMISUPERVISED_TRAINER
Semi-Supervised Adversarial Autoencoder trainer

Stefan Wong 2019
"""

import importlib
import torch
import torch.nn.functional as F
import numpy as np
from lernomatic.models import common
from lernomatic.train import trainer

#debug
#from pudb import set_trace; set_trace()


def sample_categorical(batch_size:int, n_classes:int=10) -> torch.Tensor:

    cat = np.random.randint(0, num_classes, batch_size)
    cat = np.eye(n_classes)[cat].astype('float32')
    cat = torch.from_numpy(cat)

    return torch.Tensor(cat)


class AAESemiTrainer(trainer.Trainer):
    def __init__(self,
                 q_net:common.LernomaticModel=None,
                 p_net:common.LernomaticModel=None,
                 d_cat_net:common.LernomaticModel=None,
                 d_gauss_net:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.q_net             = q_net
        self.p_net             = p_net
        self.d_cat_net         = d_cat_net
        self.d_gauss_net       = d_gauss_net
        # keyword args specific to this trainer
        self.gen_lr    : float = kwargs.pop('gen_lr', 0.006)
        self.reg_lr    : float = kwargs.pop('reg_lr', 0.0008)
        self.semi_lr   : float = kwargs.pop('semi_lr', 0.001)
        self.eps       : float = kwargs.pop('eps', 1e-15)
        self.data_norm : float = kwargs.pop('data_norm', 0.3081 + 0.1307)
        # in addition to the usual train loader, we also have a labelled train
        # loader
        self.train_label_dataset = kwargs.pop('train_label_dataset', None)
        self.val_label_dataset   = kwargs.pop('val_label_dataset', None)

        super(AAESemiTrainer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'AAESemiTrainer'

    def _init_dataloaders(self) -> None:
        super(AAESemiTrainer, self)._init_dataloaders()
        # also init the 'label' dataloaders

        if self.train_label_dataset is None:
            self.train_label_loader = None
        else:
            self.train_label_loader = torch.utils.data.DataLoader(
                self.train_label_dataset,
                batch_size = self.batch_size,
                shuffle = self.shuffle
            )

        if self.val_label_dataset is None:
            self.val_label_loader = None
        else:
            self.val_label_loader = torch.utils.data.DataLoader(
                self.val_label_dataset,
                batch_size = self.batch_size,
                shuffle = self.shuffle
            )

    def _init_optimizer(self) -> None:
        # create optimizers for each of the models

        # optimizers for unsupervised phase
        if self.p_net is not None:
            self.p_decoder_optim = torch.optim.Adam(
                self.p_net.get_model_parameters(),
                lr = self.gen_lr
            )

        if self.q_net is not None:
            self.q_encoder_optim = torch.optim.Adam(
                self.q_net.get_model_parameters(),
                lr = self.gen_lr
            )
            self.q_generator_optim = torch.optim.Adam(
                self.q_net.get_model_parameters(),
                lr = self.reg_lr
            )

        if self.d_cat_net is not None:
            self.d_cat_optim = torch.optim.Adam(
                self.d_cat_net.get_model_parameters(),
                lr = self.reg_lr
            )

        if self.d_gauss_net is not None:
            self.d_gauss_optim = torch.optim.Adam(
                self.d_gauss_net.get_model_parameters(),
                lr = self.reg_lr
            )

        # optimizers for semi-supervised
        if self.q_net is not None:
            self.q_semi_optim = torch.optim.Adam(
                self.q_net.get_model_parameters(),
                lr = self.semi_lr
            )

    def _init_history(self) -> None:
        self.loss_iter      = 0
        self.val_loss_iter  = 0
        self.acc_iter       = 0
        self.iter_per_epoch = int(len(self.train_loader) / self.num_epochs)

        self.g_loss_history       = np.zeros(len(self.train_loader) * self.num_epochs)
        self.d_cat_loss_history   = np.zeros(len(self.train_loader) * self.num_epochs)
        self.d_gauss_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)
        self.recon_loss_history   = np.zeros(len(self.train_loader) * self.num_epochs)

    def _send_to_device(self) -> None:
        if self.p_net is not None:
            self.p_net.send_to(self.device)

        if self.d_cat_net is not None:
            self.d_cat_net.send_to(self.device)

        if self.d_gauss_net is not None:
            self.d_gauss_net.send_to(self.device)

    def _zero_all_nets(self) -> None:
        self.p_net.zero_grad()
        self.q_net.zero_grad()
        self.d_cat_net.zero_grad()
        self.d_gauss_net.zero_grad()

    def _zero_all_optim(self) -> None:
        self.q_semi_optim.zero_grad()
        self.d_gauss_optim.zero_grad()
        self.d_cat_optim.zero_grad()
        self.q_gnerator_optim.zero_grad()
        self.q_encoder_optim.zero_grad()
        self.p_decoder_optim.zero_grad()

    def train_epoch(self) -> None:
        """
        TRAIN_EPOCH
        Run a single epoch of the training routine
        """
        if self.train_label_loader is None:
            raise RuntimeError('No train label loader in [%s]' % repr(self))

        self.p_net.set_train()
        self.q_net.set_train()
        self.d_cat_net.set_train()
        self.d_gauss_net.set_train()

        #for batch_idx, (data, target) in enumerate(self.train_loader):
        #    data = data.to(self.device)
        #    target = target.to(self.device)

        for batch_idx, ((X_l, target_l), (X_u, target_u)) in enumerate(zip(self.train_label_loader, self.train_loader)):
            for X, target in [(X_u, target_u), (X_l, target_l)]:
                if target[0] == -1:
                    labelled = False
                else:
                    labelled = True

                # send data to device
                X.resize_(self.batch_size, self.q_net.get_x_dim())
                X      = X.to(self.device)
                target = target.to(self.device)

                # init gradients
                self._zero_all_nets()

                # ==== Reconstruction phase ==== $
                if not labelled:
                    z_sample = torch.cat(self.q_net(X), 1)
                    x_sample = self.p_net(z_sample)

                    recon_loss = F.binary_cross_entropy(
                        x_sample + self.eps,
                        X.resize(self.batch_size, self.q_net.get_x_dim()) + self.eps
                    )
                    recon_loss.backward()
                    self.p_decoder_optim.step()
                    self.q_decoder_optim.step()
                    self._zero_all_nets()

                    # ==== Regularization Phase ==== #
                    # disciminator
                    self.q_net.set_eval()
                    z_real_cat   = sample_categorical(self.batch_size, n_classes=10)
                    z_real_gauss = torch.Tensor(torch.randn(self.batch_size, self.q_net.get_z_dim()))

                    z_real_cat   = z_real_cat.to(self.device)
                    z_real_gauss = z_real_gauss.to(self.device)

                    z_fake_cat, z_fake_gauss = self.q_net(X)

                    d_real_cat   = self.d_cat_net(z_real_cat)
                    d_real_gauss = self.d_gauss_net(z_real_cat)
                    d_fake_cat   = self.d_cat_net(z_fake_cat)
                    d_fake_gauss = self.d_gauss_net(z_fake_cat)

                    d_loss_cat = -torch.mean(torch.log(d_real_cat + self.eps), torch.log(1.0 - d_fake_cat + self.eps))
                    d_loss_gauss = -torch.mean(torch.log(d_real_gauss + self.eps) + torch.log(1.0 - d_fake_gauss + self.eps))

                    d_loss = d_loss_cat + d_loss_gauss
                    d_loss.backward()

                    self.d_cat_optim.step()
                    self.d_gauss_optim.step()
                    self._zero_all_nets()

                    # ==== Generator ==== #
                    self.q_net.set_train()
                    z_fake_cat, z_fake_gauss = self.q_net(X)

                    d_fake_cat = self.d_cat_net(z_fake_cat)
                    d_fake_gauss = self.d_gauss_net(z_fake_cat)

                    g_loss = -torch.mean(torch.log(d_fake_cat + self.eps)) - torch.mean(torch.log(d_fake_gauss + self.eps))
                    g_loss.backward()
                    self.q_decoder_optim.step()
                    self._zero_all_nets()

                # ==== Semi-supervised phase ==== #
                if labelled:
                    pred, _ = self.q_net(X)
                    class_loss = F.cross_entropy(pred, target)
                    class_loss.backward()
                    self.q_semi_optim.step()

                    self._zero_all_nets()

            # display
            if (batch_idx > 0) and (batch_idx % self.print_every) == 0:
                if labelled:
                    print('[TRAIN] :   Epoch       iteration  [labelled]    Class Loss ')
                    print('            [%3d/%3d]   [%6d/%6d]             %.6f   ' %\
                            (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_loader),
                            class_loss.item())
                    )
                else:
                    print('[TRAIN] :   Epoch       iteration  [unlabelled] G Loss    D Loss     R Loss')
                    print('            [%3d/%3d]   [%6d/%6d]  %.6f   %.6f   %.6f' %\
                            (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_loader),
                            g_loss.item(), d_loss.item(), recon_loss.item() )
                    )

