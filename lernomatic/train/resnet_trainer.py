"""
RESNET_TRAINER
Trainer for Resnet models

Stefan Wong 2019
"""

import importlib
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from lernomatic.train import trainer
from lernomatic.models import resnets
from lernomatic.models import common

class ResnetTrainer(trainer.Trainer):
    """
    ResnetTrainer

    Trainer object for resnet experiments. This trainer is almost no different from the
    default Trainer object save for the fact that it loads the CIFAR-10 dataset by
    default.
    """
    def __init__(self, model: common.LernomaticModel, **kwargs) -> None:
        self.data_dir      = kwargs.pop('data_dir', 'data/')
        self.augment_data  = kwargs.pop('augment_data', False)
        self.train_dataset = kwargs.pop('train_dataset', None)
        self.test_dataset  = kwargs.pop('test_dataset', None)
        super(ResnetTrainer, self).__init__(model, **kwargs)

        self.criterion = torch.nn.CrossEntropyLoss()

    def __repr__(self) -> str:
        return 'ResnetTrainer'

    def __str__(self) -> str:
        s = []
        s.append('ResnetTrainer \n')
        if self.train_loader is not None:
            s.append('Training set size :%d\n' % len(self.train_loader.dataset))
        else:
            s.append('Training set not loaded\n')

        if self.val_loader is not None:
            s.append('Validation set size :%d\n' % len(self.val_loader.dataset))
        else:
            s.append('Validation set not loaded\n')

        return ''.join(s)

    def _init_dataloaders(self) -> None:
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

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        # init datasets- use CIFAR10 if we don't specify otherwise
        if self.train_dataset is None:
            self.train_dataset = torchvision.datasets.CIFAR10(
                self.data_dir,
                train=True,
                download=True,
                transform=train_transform
            )
        if self.val_dataset is None:
            self.val_dataset = torchvision.datasets.CIFAR10(
                self.data_dir,
                train=False,
                download=True,
                transform=val_transforms
            )

        # init loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size  = self.batch_size,
            shuffle     = self.shuffle,
            num_workers = self.num_workers
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size  = self.batch_size,      # <- NOTE: was self.val_batch_size
            shuffle     = self.shuffle,
            num_workers = self.num_workers
        )

        self.test_loader = None

    def save_history(self, fname: str) -> None:
        history = dict()
        history['loss_history']      = self.loss_history
        history['loss_iter']         = self.loss_iter
        history['val_loss_history']  = self.val_loss_history
        history['val_loss_iter']     = self.val_loss_iter
        history['acc_history']       = self.acc_history
        history['acc_iter']          = self.acc_iter
        history['cur_epoch']         = self.cur_epoch
        history['iter_per_epoch']    = self.iter_per_epoch
        if self.val_loss_history is not None:
            history['val_loss_history'] = self.val_loss_history

        torch.save(history, fname)

    def load_history(self, fname: str) -> None:
        history = torch.load(fname)
        self.loss_history      = history['loss_history']
        self.loss_iter         = history['loss_iter']
        self.val_loss_history  = history['val_loss_history']
        self.val_loss_iter     = history['val_loss_iter']
        self.acc_history       = history['acc_history']
        self.acc_iter          = history['acc_iter']
        self.cur_epoch         = history['cur_epoch']
        self.iter_per_epoch    = history['iter_per_epoch']
        if 'val_loss_history' in history:
            self.val_loss_history = history['val_loss_history']

    # For resnets, we need to pass the correct depth parameter in first
    def load_checkpoint(self, fname: str) -> None:
        """
        Load all data from a checkpoint
        """
        checkpoint_data = torch.load(fname, map_location='cpu')
        self.set_trainer_params(checkpoint_data['trainer_params'])
        # here we just load the object that derives from LernomaticModel. That
        # object will in turn load the actual nn.Module data from the
        # checkpoint data with the 'model' key
        model_import_path = checkpoint_data['model']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['model']['model_name'])
        self.model = mod(
            depth          = checkpoint_data['model']['resnet_params']['depth'],
            num_classes    = checkpoint_data['model']['resnet_params']['num_classes'],
            input_channels = checkpoint_data['model']['resnet_params']['input_channels'],
            w_factor       = checkpoint_data['model']['resnet_params']['w_factor'],
            drop_rate      = checkpoint_data['model']['resnet_params']['drop_rate']
        )
        self.model.set_params(checkpoint_data['model'])

        # Load optimizer
        self._init_optimizer()
        self.optimizer.load_state_dict(checkpoint_data['optim'])
        # Transfer all the tensors to the current device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        # restore trainer object info
        self._send_to_device()
