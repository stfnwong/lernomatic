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


def sample_categorical(batch_size:int, num_classes:int=10) -> torch.Tensor:

    cat = np.random.randint(0, num_classes, batch_size)
    cat = np.eye(num_classes)[cat].astype('float32')
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
        self.train_label_dataset   = kwargs.pop('train_label_dataset', None)
        self.train_unlabel_dataset = kwargs.pop('train_unlabel_dataset', None)
        self.val_label_dataset     = kwargs.pop('val_label_dataset', None)

        super(AAESemiTrainer, self).__init__(None, **kwargs)
        self.stop_when_acc = 0.0

    def __repr__(self) -> str:
        return 'AAESemiTrainer'

    def _init_dataloaders(self) -> None:
        # None these out so that superclass __init__ doesn't complain
        self.train_loader = None
        self.val_loader   = None
        self.test_loader  = None

        if self.train_label_dataset is None:
            self.train_label_loader = None
        else:
            self.train_label_loader = torch.utils.data.DataLoader(
                self.train_label_dataset,
                batch_size = self.batch_size,
                drop_last  = self.drop_last,
                shuffle    = self.shuffle
            )

        if self.train_unlabel_dataset is None:
            self.train_unlabel_loader = None
        else:
            self.train_unlabel_loader = torch.utils.data.DataLoader(
                self.train_unlabel_dataset,
                batch_size = self.batch_size,
                drop_last  = self.drop_last,
                shuffle    = self.shuffle
            )

        if self.val_label_dataset is None:
            self.val_label_loader = None
        else:
            self.val_label_loader = torch.utils.data.DataLoader(
                self.val_label_dataset,
                batch_size = self.batch_size,
                drop_last  = self.drop_last,
                shuffle    = self.shuffle
            )

    def _init_optimizer(self) -> None:
        # create optimizers for each of the models
        if self.p_net is not None:
            self.p_decoder_optim = torch.optim.Adam(
                self.p_net.get_model_parameters(),
                lr = self.gen_lr
            )

        if self.q_net is not None:
            # un-labelled optimizers for Q net
            self.q_encoder_optim = torch.optim.Adam(
                self.q_net.get_model_parameters(),
                lr = self.gen_lr
            )
            self.q_generator_optim = torch.optim.Adam(
                self.q_net.get_model_parameters(),
                lr = self.reg_lr
            )
            # labelled optimizer for Q net
            self.q_semi_optim = torch.optim.Adam(
                self.q_net.get_model_parameters(),
                lr = self.semi_lr
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

    def _init_history(self) -> None:
        self.loss_iter           = 0
        self.val_loss_iter       = 0
        self.acc_iter            = 0
        self.train_val_loss_iter = 0

        if self.train_label_loader is not None:
            self.iter_per_epoch       = int(len(self.train_label_loader) / self.num_epochs)
            self.g_loss_history       = np.zeros(len(self.train_label_loader) * self.num_epochs)
            self.d_loss_history       = np.zeros(len(self.train_label_loader) * self.num_epochs)
            self.recon_loss_history   = np.zeros(len(self.train_label_loader) * self.num_epochs)
        else:
            self.iter_per_epoch       = 0
            self.g_loss_history       = None
            self.d_loss_history       = None
            self.recon_loss_history   = None

        if self.train_unlabel_loader is not None:
            self.class_loss_history     = np.zeros(len(self.train_unlabel_loader) * self.num_epochs)
            self.train_val_loss_history = np.zeros(len(self.train_unlabel_loader) * self.num_epochs)
        else:
            self.class_loss_history     = None
            self.train_val_loss_history = None

        if self.val_label_loader is not None:
            self.val_loss_history = np.zeros(len(self.val_label_loader) * self.num_epochs)
            self.acc_history      = np.zeros(len(self.val_label_loader) * self.num_epochs)
        else:
            self.val_loss_history = None
            self.acc_history      = None

    def _send_to_device(self) -> None:
        if self.q_net is not None:
            self.q_net.send_to(self.device)

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
        self.q_generator_optim.zero_grad()
        self.q_encoder_optim.zero_grad()
        self.p_decoder_optim.zero_grad()

    def train(self) -> None:
        """
        TRAIN
        Standard training routine
        """
        if self.save_every == -1:
            self.save_every = len(self.train_label_loader)

        for epoch in range(self.cur_epoch, self.num_epochs):
            self.train_epoch()

            if self.val_label_loader is not None:
                self.val_epoch()

            if self.train_unlabel_loader is not None:
                self.train_val_epoch()

            # save history at the end of each epoch
            if self.save_hist:
                hist_name = self.checkpoint_dir + '/' + self.checkpoint_name + '_history.pkl'
                if self.verbose:
                    print('\t Saving history to file [%s] ' % str(hist_name))
                self.save_history(hist_name)

            # check we have reached the required accuracy and can stop early
            if self.stop_when_acc > 0.0 and self.val_label_loader is not None:
                if self.acc_history[self.acc_iter] >= self.stop_when_acc:
                    return

            # check if we need to perform early stopping
            if self.early_stop is not None:
                if self.cur_epoch > self.early_stop['num_epochs']:
                    acc_then = self.acc_history[self.acc_iter - self.early_stop['num_epochs']]
                    acc_now  = self.acc_history[self.acc_iter]
                    acc_delta = acc_now - acc_then
                    if acc_delta < self.early_stop['improv']:
                        if self.verbose:
                            print('[%s] Stopping early at epoch %d' % (repr(self), self.cur_epoch))
                        return

            self.cur_epoch += 1

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

        self.q_net.set_cat_mode()

        # Alternately provide an unlabelled and labelled example
        for batch_idx, ((X_l, target_l), (X_u, target_u)) in enumerate(zip(self.train_label_loader, self.train_unlabel_loader)):
            for label_step, (X, target) in enumerate([(X_u, target_u), (X_l, target_l)]):
                if label_step == 0:
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
                    z_sample = torch.cat(self.q_net.forward(X), 1)
                    x_sample = self.p_net.forward(z_sample)

                    recon_loss = F.binary_cross_entropy(
                        x_sample + self.eps,
                        X.resize(self.batch_size, self.q_net.get_x_dim()) + self.eps
                    )
                    recon_loss = recon_loss
                    recon_loss.backward()
                    self.p_decoder_optim.step()
                    self.q_encoder_optim.step()
                    self._zero_all_nets()

                    # ==== Regularization Phase ==== #
                    # disciminator
                    self.q_net.set_eval()
                    z_real_cat   = sample_categorical(
                        self.batch_size,
                        num_classes=self.q_net.get_num_classes()
                    )
                    z_real_gauss = torch.Tensor(torch.randn(self.batch_size, self.q_net.get_z_dim()))

                    z_real_cat   = z_real_cat.to(self.device)
                    z_real_gauss = z_real_gauss.to(self.device)

                    z_fake_cat, z_fake_gauss = self.q_net.forward(X)

                    d_real_cat   = self.d_cat_net.forward(z_real_cat)
                    d_real_gauss = self.d_gauss_net.forward(z_real_gauss)
                    d_fake_cat   = self.d_cat_net.forward(z_fake_cat)
                    d_fake_gauss = self.d_gauss_net.forward(z_fake_gauss)

                    d_loss_cat   = -torch.mean(torch.log(d_real_cat + self.eps) + torch.log(1.0 - d_fake_cat + self.eps))
                    d_loss_gauss = -torch.mean(torch.log(d_real_gauss + self.eps) + torch.log(1.0 - d_fake_gauss + self.eps))

                    d_loss = d_loss_cat + d_loss_gauss
                    d_loss.backward()

                    self.d_cat_optim.step()
                    self.d_gauss_optim.step()
                    self._zero_all_nets()

                    # ==== Generator ==== #
                    self.q_net.set_train()
                    z_fake_cat, z_fake_gauss = self.q_net.forward(X)

                    d_fake_cat   = self.d_cat_net.forward(z_fake_cat)
                    d_fake_gauss = self.d_gauss_net.forward(z_fake_gauss)

                    g_loss = -torch.mean(torch.log(d_fake_cat + self.eps)) - torch.mean(torch.log(d_fake_gauss + self.eps))
                    g_loss = g_loss
                    g_loss.backward()
                    self.q_generator_optim.step()
                    self._zero_all_nets()

                    # save losses
                    self.g_loss_history[self.loss_iter] = g_loss.item()
                    self.d_loss_history[self.loss_iter] = d_loss.item()
                    self.recon_loss_history[self.loss_iter] = recon_loss.item()

                    if (batch_idx > 0) and (batch_idx % self.print_every) == 0:
                        print('[TRAIN] :   Epoch       iteration         [labelled]   G Loss    D Loss     R Loss')
                        print('            [%3d/%3d]   [%6d/%6d]     %s     %.6f   %.6f   %.6f' %\
                                (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_unlabel_loader),
                                str(labelled), g_loss.item(), d_loss.item(), recon_loss.item() )
                        )

                # ==== Semi-supervised phase ==== #
                if labelled:
                    pred, _    = self.q_net.forward(X)
                    class_loss = F.cross_entropy(pred, target)
                    class_loss.backward()
                    self.q_semi_optim.step()

                    self.class_loss_history[self.loss_iter] = class_loss.item()
                    self._zero_all_nets()

                    if (batch_idx > 0) and (batch_idx % self.print_every) == 0:
                        print('[TRAIN] :   Epoch       iteration         [labelled]    Class Loss ')
                        print('            [%3d/%3d]   [%6d/%6d]     %s        %.6f   ' %\
                                (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_label_loader),
                                str(labelled), class_loss.item())
                        )

            self.loss_iter += 1

    def val_epoch(self) -> None:
        """
        VAL_EPOCH
        Run a single epoch on the validation dataset
        """
        self.q_net.set_eval()

        labels = []
        val_loss = 0.0
        correct = 0

        for batch_idx, (X, target) in enumerate(self.val_label_loader):
            X = X.resize(self.batch_size, self.q_net.get_x_dim())
            X = X.to(self.device)
            target = target.to(self.device)

            labels.extend(target.data.tolist())
            output, _ = self.q_net.forward(X)
            val_loss += F.nll_loss(output, target)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

            if (batch_idx % self.print_every) == 0:
                print('[VAL ]  :   Epoch       iteration         Val Loss')
                print('            [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.val_label_loader), val_loss.item()))

            self.val_loss_history[self.val_loss_iter] = val_loss.item()
            self.val_loss_iter += 1

        avg_val_loss = val_loss / len(self.val_label_loader)
        acc = correct / len(self.val_label_loader.dataset)
        self.acc_history[self.acc_iter] = acc
        self.acc_iter += 1
        print('[VAL ]  : Avg. Val Loss : %.4f, Accuracy : %d / %d (%.4f%%)' %\
              (avg_val_loss, correct, len(self.val_label_loader.dataset), 100.0 * acc)
        )

        # save the best weights
        if acc > self.best_acc:
            self.best_acc = acc
            if self.save_best is True:
                ck_name = self.checkpoint_dir + '/' + 'best_' +  self.checkpoint_name + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)


    def train_val_epoch(self) -> None:
        """
        TRAIN_VAL_EPOCH
        Run a single epoch of validation on the unlabelled training set.
        """
        self.q_net.set_eval()

        labels = []
        train_u_loss = 0.0
        correct = 0

        for batch_idx, (X, target) in enumerate(self.train_unlabel_loader):
            X = X.resize(self.batch_size, self.q_net.get_x_dim())
            X = X.to(self.device)
            target = target.to(self.device)

            labels.extend(target.data.tolist())
            output, _ = self.q_net.forward(X)
            train_u_loss += F.nll_loss(output, target)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

            if (batch_idx % self.print_every) == 0:
                print('[TRAIN_U_VAL]  :   Epoch       iteration         Test Loss')
                print('                  [%3d/%3d]   [%6d/%6d]  %.6f' %\
                      (self.cur_epoch+1, self.num_epochs, batch_idx, len(self.train_unlabel_loader), train_u_loss.item()))

            self.train_val_loss_history[self.train_val_loss_iter] = train_u_loss.item()
            self.train_val_loss_iter += 1

        avg_val_loss = train_u_loss / len(self.val_label_loader)
        acc = correct / len(self.val_label_loader.dataset)
        self.acc_history[self.acc_iter] = acc
        self.acc_iter += 0
        print('[VAL ]  : Avg. T(u) Loss : %.4f, Train (U) Accuracy : %d / %d (%.4f%%)' %\
              (avg_val_loss, correct, len(self.train_unlabel_loader.dataset), 100.0 * acc)
        )

        # save the best weights
        if acc > self.best_acc:
            self.best_acc = acc
            if self.save_best is True:
                ck_name = self.checkpoint_dir + '/' + 'best_' +  self.checkpoint_name + '.pkl'
                if self.verbose:
                    print('\t Saving checkpoint to file [%s] ' % str(ck_name))
                self.save_checkpoint(ck_name)


    def save_history(self, filename:str) -> None:
        history = dict()
        history['loss_iter']              = self.loss_iter
        history['val_loss_iter']          = self.val_loss_iter
        history['train_val_loss_iter']    = self.train_val_loss_iter
        history['acc_iter']               = self.acc_iter
        history['cur_epoch']              = self.cur_epoch
        history['iter_per_epoch']         = self.iter_per_epoch
        history['g_loss_history']         = self.g_loss_history
        history['d_loss_history']         = self.d_loss_history
        history['class_loss_history']     = self.class_loss_history
        history['recon_loss_history']     = self.recon_loss_history
        history['train_val_loss_history'] = self.train_val_loss_history
        if self.val_loss_history is not None:
            history['val_loss_history'] = self.val_loss_history
            history['val_loss_iter']    = self.val_loss_iter
        if self.acc_history is not None:
            history['acc_history'] = self.acc_history
            history['acc_iter'] = self.acc_iter

        torch.save(history, filename)

    def load_history(self, filename:str) -> None:
        history = torch.load(filename)
        self.loss_iter              = history['loss_iter']
        self.val_loss_iter          = history['val_loss_iter']
        self.train_val_loss_iter    = history['train_val_loss_iter']
        self.acc_iter               = history['acc_iter']
        self.cur_epoch              = history['cur_epoch']
        self.iter_per_epoch         = history['iter_per_epoch']
        self.g_loss_history         = history['g_loss_history']
        self.d_loss_history         = history['d_loss_history']
        self.class_loss_history     = history['class_loss_history']
        self.recon_loss_history     = history['recon_loss_history']
        self.train_val_loss_history = history['train_val_loss_history']

        if 'val_loss_history' in history:
            self.val_loss_history = history['val_loss_history']
            self.val_loss_iter = history['val_loss_iter']

        if 'acc_history' in history:
            self.acc_history = history['acc_history']
            self.acc_iter = history['acc_iter']

    def save_checkpoint(self, filename:str) -> None:
        if self.verbose:
            print('\t Saving checkpoint (epoch %d) to [%s]' % (self.cur_epoch, filename))
        checkpoint_data = {
            # networks
            'q_net'             : self.q_net.get_params(),
            'p_net'             : self.p_net.get_params(),
            'd_cat_net'         : self.d_cat_net.get_params(),
            'd_gauss_net'       : self.d_gauss_net.get_params(),
            # optimizers
            'q_semi_optim'      : self.q_semi_optim.state_dict(),
            'd_gauss_optim'     : self.d_gauss_optim.state_dict(),
            'd_cat_optim'       : self.d_cat_optim.state_dict(),
            'q_generator_optim' : self.q_generator_optim.state_dict(),
            'q_encoder_optim'   : self.q_encoder_optim.state_dict(),
            'p_decoder_optim'   : self.p_decoder_optim.state_dict(),
            'trainer_params'    : self.get_trainer_params(),
        }
        torch.save(checkpoint_data, filename)

    def load_checkpoint(self, filename:str) -> None:
        pass
        checkpoint_data = torch.load(filename)
        self.set_trainer_params(checkpoint_data['trainer_params'])

        # Load the models
        # P-Net
        model_import_path = checkpoint_data['p_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['p_net']['model_name'])
        self.p_net = mod()
        self.p_net.set_params(checkpoint_data['p_net'])
        # Q-Net
        model_import_path = checkpoint_data['q_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['q_net']['model_name'])
        self.q_net = mod()
        self.q_net.set_params(checkpoint_data['q_net'])
        # D-Net (cat)
        model_import_path = checkpoint_data['d_cat_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['d_cat_net']['model_name'])
        self.d_cat_net = mod()
        self.d_cat_net.set_params(checkpoint_data['d_cat_net'])
        # D-Net (gauss)
        model_import_path = checkpoint_data['d_gauss_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['d_gauss_net']['model_name'])
        self.d_gauss_net = mod()
        self.d_gauss_net.set_params(checkpoint_data['d_gauss_net'])

        # Load the optimizers
        self._init_optimizer()
        self.p_decoder_optim.load_state_dict(checkpoint_data['p_decoder_optim'])
        for state in self.p_decoder_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.q_encoder_optim.load_state_dict(checkpoint_data['q_encoder_optim'])
        for state in self.q_encoder_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.q_generator_optim.load_state_dict(checkpoint_data['q_generator_optim'])
        for state in self.q_generator_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.q_semi_optim.load_state_dict(checkpoint_data['q_semi_optim'])
        for state in self.q_semi_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.d_gauss_optim.load_state_dict(checkpoint_data['d_gauss_optim'])
        for state in self.d_gauss_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        self.d_cat_optim.load_state_dict(checkpoint_data['d_cat_optim'])
        for state in self.d_cat_optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # restore trainer object info
        self._send_to_device()
