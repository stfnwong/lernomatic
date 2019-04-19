"""
DCGAN_TRAINER
Trainer module for DCGANs

Stefan Wong 2019
"""

import importlib
import torch
import torch.nn as nn
import numpy as np
from lernomatic.train import trainer
from lernomatic.models import dcgan
from lernomatic.models import common

# debug
#from pudb import set_trace; set_trace()


class DCGANTrainer(trainer.Trainer):
    def __init__(self, D, G, **kwargs) -> None:
        self.discriminator = D
        self.generator     = G
        self.beta1         = kwargs.pop('beta1', 0.5)
        self.real_label    = kwargs.pop('real_label', 1)
        self.fake_label    = kwargs.pop('fake_label', 0)
        self.train_loader  = None        # so that super() does not call _init_history()
        super(DCGANTrainer, self).__init__(None, **kwargs)

        self.train_dataset = kwargs.pop('train_dataset', None)
        self.test_dataset  = kwargs.pop('test_dataset', None)
        self.val_dataset   = kwargs.pop('val_dataset', None)

        # use CELoss
        self.loss_function = 'BCELoss'
        self.optim_function = 'Adam'
        self.criterion = nn.BCELoss()

        # setup internals
        self._init_device()
        self._init_dataloaders()
        self._init_optimizers()
        self._init_history()
        self._send_to_device();

    def __repr__(self) -> str:
        return 'DCGANTrainer'

    def _init_dataloaders(self) -> None:
        if self.train_dataset is None:
            self.train_loader = None
        else:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                shuffle = self.shuffle,
                drop_last = True
            )
        self.test_loader = None     # TODO : clean up references to this later

    def _init_history(self) -> None:
        self.loss_iter = 0
        self.test_loss_iter = 0
        self.acc_iter = 0
        self.iter_per_epoch = int(len(self.train_loader) / self.num_epochs)
        self.d_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)
        self.g_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)

    def _init_optimizers(self) -> None:
        # generator
        if self.generator is None:
            self.optim_g = None
        else:
            self.optim_g = torch.optim.Adam(
                self.generator.get_model_parameters(),
                lr = self.learning_rate,
                betas = (self.beta1, 0.999)
            )
        # discriminator
        if self.discriminator is None:
            self.optim_d = None
        else:
            self.optim_d = torch.optim.Adam(
                self.discriminator.get_model_parameters(),
                lr = self.learning_rate,
                betas = (self.beta1, 0.999)
            )

    def _send_to_device(self) -> None:
        if self.generator is not None:
            self.generator.send_to(self.device)
        if self.discriminator is not None:
            self.discriminator.send_to(self.device)

    def _weight_init(self,  model:nn.Module) -> None:
        # The orignal paper suggests that the weights should be initialized
        # randomly from a normal distribution with mean=0, std=0.02
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)

    def set_discriminator(self, D:common.LernomaticModel) -> None:
        self.discriminator = D

    def set_generator(self, G:common.LernomaticModel) -> None:
        self.generator = G

    def get_trainer_params(self) -> dict:
        params = {
            # labels
            'fake_label' : self.fake_label,
            'real_label' : self.real_label,
            'beta1'      : self.beta1,
        }
        params.update(super(DCGANTrainer, self).get_trainer_params())
        return params

    def set_trainer_params(self, params:dict) -> None:
        super(DCGANTrainer, self).set_trainer_params(params)
        # set the subclass params
        self.fake_label = params['fake_label']
        self.real_label = params['real_label']
        self.beta1      = params['beta1']

    def gen_fixed_noise_vector(self, vec_dim:int) -> None:
        self.fixed_noise = torch.randn(64, vec_dim, 1, 1, device=self.device)

    # ==== TRAINING ==== #
    def train_epoch(self) -> None:
        self.discriminator.set_train()
        self.generator.set_train()

        for n, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            # Update D NETWORK
            # Maximum log(D(x)) + log(1 - D(G(z)))
            self.discriminator.zero_grad()

            # make a tensor of real labels
            label = torch.full((self.batch_size,), self.real_label, device=self.device)
            # compute loss on all-real batch
            real_output = self.discriminator.forward(data)
            errd_real = self.criterion(real_output, label)
            errd_real.backward()

            d_x = real_output.mean().item()

            # Now train with an all-fake batch
            noise = torch.randn(
                self.batch_size,
                self.generator.get_zvec_dim(),
                1, 1,
                device = self.device
            )
            # Update G network (maximize log(D(G(z))))
            fake = self.generator.forward(noise)
            label.fill_(self.fake_label)
            # classify all the fake batches with D. Tensor is flattened here
            disc_output = self.discriminator.forward(fake.detach()).view(-1)
            # compute D's loss on the all fake batch
            errd_fake = self.criterion(disc_output, label)
            errd_fake.backward()
            # compute gradients for this batch
            dg_z1 = disc_output.mean().item()
            err_d = errd_real + errd_fake

            self.optim_d.step()

            # Update the Generator network
            self.generator.zero_grad()
            label.fill_(self.real_label)
            output = self.discriminator.forward(fake).view(-1)
            # compute G's loss based on the output from D
            err_g = self.criterion(output, label)
            # compute gradients for G and update
            err_g.backward()
            dg_z2 = output.mean().item()
            self.optim_g.step()

            # save history
            self.d_loss_history[self.loss_iter] = err_d.item()
            self.g_loss_history[self.loss_iter] = err_g.item()
            self.loss_iter += 1

            # display
            if self.print_every > 0 and (self.loss_iter % self.print_every) == 0:
                print('[TRAIN] :   Epoch      iteration      Loss (G)   Loss (D)')
                print('          [%3d/%3d]  [%6d/%6d]  %.6f   %.6f  ' %\
                      (self.cur_epoch+1, self.num_epochs, n, len(self.train_loader), err_d.item(), err_g.item())
                )
                print('[TRAIN] : D(x)     D(G(z)) [1/2]')
                print('          %.4f   %.4f / %.4f' % (d_x, dg_z2, dg_z1))

            # save
            if self.save_every > 0 and (self.loss_iter % self.save_every) == 0:
                ck_name = self.checkpoint_dir + self.checkpoint_name + '_iter_' + str(self.loss_iter) +\
                    '_epoch_' + str(self.cur_epoch) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s]' % str(ck_name))
                self.save_checkpoint(ck_name)


    def train(self) -> None:
        self._send_to_device()
        self.gen_fixed_noise_vector(self.generator.get_zvec_dim())

        for n in range(self.cur_epoch, self.num_epochs):
            self.train_epoch()
            # since this is an unsupervised task we don't have a good notion of
            # 'accuracy', therefore no test phase
            # save history at the end of each epoch
            hist_name = self.checkpoint_dir + '/' + self.checkpoint_name + '_history.pkl'
            if self.verbose:
                print('\t Saving history to file [%s] ' % str(hist_name))
            self.save_history(hist_name)

            self.cur_epoch += 1


    # also need to overload some of the history functions
    def get_loss_history(self) -> tuple:        # TODO ; more extensive type hint?
        return (self.g_loss_history[0 : self.loss_iter], self.d_loss_history[0 : self.loss_iter])

    def get_g_loss_history(self) -> np.ndarray:
        return self.g_loss_history[0 : self.loss_iter]

    def get_d_loss_history(self) -> np.ndarray:
        return self.d_loss_history[0 : self.loss_iter]

    def get_test_loss_history(self) -> None:
        return None

    # Checkpointing
    def save_checkpoint(self, fname:str) -> None:
        checkpoint = {
            'trainer_params' : self.get_trainer_params(),
            'discriminator'  : self.discriminator.get_params(),
            'generator'      : self.generator.get_params(),
            'optim_d'        : self.optim_d.state_dict(),
            'optim_g'        : self.optim_g.state_dict()
        }
        torch.save(checkpoint, fname)

    def load_checkpoint(self, fname: str) -> None:
        checkpoint_data = torch.load(fname)
        # here we just load the object that derives from LernomaticModel. That
        # object will in turn load the actual nn.Module data from the
        # checkpoint data with the 'model' key

        # load generator
        gen_import_path = checkpoint_data['generator']['model_import_path']
        gen_imp = importlib.import_module(gen_import_path)
        gen = getattr(gen_imp, checkpoint_data['generator']['model_name'])
        self.generator = gen()
        self.generator.set_params(checkpoint_data['generator'])

        # load discriminator
        dis_import_path = checkpoint_data['discriminator']['model_import_path']
        dis_imp = importlib.import_module(gen_import_path)
        dis = getattr(dis_imp, checkpoint_data['discriminator']['model_name'])
        self.discriminator = dis()
        self.discriminator.set_params(checkpoint_data['discriminator'])

        # Load optimizer
        self._init_optimizers()  # TODO : should this name always be plural for consistency?
        self.optim_d.load_state_dict(checkpoint_data['optim_d'])
        self.optim_g.load_state_dict(checkpoint_data['optim_g'])
        # transfer tensors to current device
        for state in self.optim_d.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        for state in self.optim_g.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # restore trainer object info
        self.set_trainer_params(checkpoint_data['trainer_params'])
        self._send_to_device()

    # History
    def save_history(self, fname:str) -> None:
        history = {
            'd_loss_history' : self.d_loss_history,
            'g_loss_history' : self.g_loss_history,
            'loss_iter'      : self.loss_iter
        }
        torch.save(history, fname)

    def load_history(self, fname:str) -> None:
        history = torch.load(fname)
        self.d_loss_history = history['d_loss_history']
        self.g_loss_history = history['g_loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch = self.start_epoch
