"""
DCGAN_TRAINER
Trainer module for DCGANs

Stefan Wong 2019
"""

import importlib
import torch
import torch.nn as nn
import torchvision
import numpy as np
from lernomatic.train import trainer
from lernomatic.models import common
from lernomatic.models.gan import dcgan

# type stuff
from typing import Tuple



class DCGANTrainer(trainer.Trainer):
    def __init__(self,
                 D:common.LernomaticModel=None,
                 G:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.discriminator :common.LernomaticModel = D
        self.generator     :common.LernomaticModel = G
        self.beta1         :float = kwargs.pop('beta1', 0.5)
        self.real_label    :int   = kwargs.pop('real_label', 1)
        self.fake_label    :int   = kwargs.pop('fake_label', 0)

        super(DCGANTrainer, self).__init__(None, **kwargs)
        # use CELoss
        self.loss_function = 'BCELoss'
        self.optim_function = 'Adam'
        self.criterion = nn.BCELoss()

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
        self.test_loader = None
        self.val_loader = None

    def _init_history(self) -> None:
        self.loss_iter = 0
        self.test_loss_iter = 0
        self.acc_iter = 0
        self.iter_per_epoch = int(len(self.train_loader) / self.num_epochs)
        self.d_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)
        self.g_loss_history = np.zeros(len(self.train_loader) * self.num_epochs)

    def _init_optimizer(self) -> None:
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

    def set_discriminator(self, D:common.LernomaticModel) -> None:
        self.discriminator = D

    def set_generator(self, G:common.LernomaticModel) -> None:
        self.generator = G

    def set_num_epochs(self, num_epochs:int) -> None:
        if num_epochs > self.num_epochs:
            # save temporary history
            temp_g_loss_history = np.copy(self.g_loss_history)
            temp_d_loss_history = np.copy(self.d_loss_history)
            temp_loss_iter      = self.loss_iter
            temp_iter_per_epoch = self.iter_per_epoch
            temp_cur_epoch      = self.cur_epoch
            self.num_epochs     = num_epochs
            self._init_history()
            # restore old history
            self.g_loss_history[:len(temp_g_loss_history)] = temp_g_loss_history
            self.d_loss_history[:len(temp_d_loss_history)] = temp_d_loss_history
            self.loss_iter      = temp_loss_iter
            self.cur_epoch      = temp_cur_epoch
            self.iter_per_epoch = temp_iter_per_epoch
        else:
            self.num_epochs = num_epochs

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

    def set_learning_rate(self, lr: float, param_zero:bool=True) -> None:
        if param_zero:
            self.optim_d.param_groups[0]['lr'] = lr
            self.optim_g.param_groups[0]['lr'] = lr
        else:
            for g in self.optim_d.param_groups:
                g['lr'] = lr

            for g in self.optim_g.param_groups:
                g['lr'] = lr

    # ==== TRAINING ==== #
    def train_epoch(self) -> None:
        self.discriminator.set_train()
        self.generator.set_train()

        for n, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)
            # Update D NETWORK
            # Maximize log(D(x)) + log(1 - D(G(z)))
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
            # optimize discriminator
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
                print('          %.4f      %.4f / %.4f' % (d_x, dg_z2, dg_z1))

                if self.tb_writer is not None:
                    self.tb_writer.add_scalar('generator/loss',     err_g.item(), self.loss_iter)
                    self.tb_writer.add_scalar('discriminator/loss', err_d.item(), self.loss_iter)
                    self.tb_writer.add_scalar('discriminator/x',    d_x,          self.loss_iter)
                    self.tb_writer.add_scalar('discriminator/gx',   dg_z2,        self.loss_iter)
                    self.tb_writer.add_scalar('discriminator/gx2',  dg_z1,        self.loss_iter)

            # save
            if self.save_every > 0 and (self.loss_iter % self.save_every) == 0:
                ck_name = self.checkpoint_dir + self.checkpoint_name + \
                    '_epoch_' + str(self.cur_epoch) + '_iter_' + str(self.loss_iter) + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s]' % str(ck_name))
                self.save_checkpoint(ck_name)

        if self.lr_scheduler is not None:
            self.apply_lr_schedule()

        # If we have a summary writer then show some example images
        if self.tb_writer is not None:
            self.generator.set_eval()

            zvec = torch.randn(self.batch_size, self.generator.get_zvec_dim(), 1, 1)
            zvec = zvec.to(self.device)
            fake = self.generator.forward(zvec).detach()
            fake = fake.to('cpu')

            grid = torchvision.utils.make_grid(fake)
            self.tb_writer.add_image('dcgan/generated', grid, self.cur_epoch)

    def val_epoch(self) -> None:
        pass

    # also need to overload some of the history functions
    def get_loss_history(self) -> Tuple[np.ndarray, np.ndarray]:
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
        self._init_optimizer()
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
            'loss_iter'      : self.loss_iter,
            'iter_per_epoch' : self.iter_per_epoch,
            'cur_epoch'      : self.cur_epoch
        }
        torch.save(history, fname)

    def load_history(self, fname:str) -> None:
        history = torch.load(fname)
        self.d_loss_history = history['d_loss_history']
        self.g_loss_history = history['g_loss_history']
        self.loss_iter      = history['loss_iter']
        self.cur_epoch      = history['cur_epoch']
