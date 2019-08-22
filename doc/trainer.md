# TRAINER 

This document details the operation of the `Trainer` module. `Trainer` is both the base class for all trainer objects in `lernomatic` as well as a stand-alone object that can be used to train simple models. 

## Trainer purpose
A `LernomaticModel` is conceptually just a collection of weights and a `forward()` function. While there are typically some utility methods attached to a `LernomaticModel`, the `lernomatic` framework generally consideres the model to just represent the computation graph alone. The specifics of the optimization routine are therefore handled inside the `Trainer` object.

The basic `Trainer` provides enough machinery to perform single-label classification tasks. More specialized trainers can subclass the `Trainer` object and override the `train_epoch()` and `val_epoch()` methods to implement things like Autoencoders, GANs, Seq2Seq models, and so on.




## Trainer structure
At minimum a `Trainer` requires
- A model object
- An optimizer drawn from `torch.optim`
- A loss function drawn from `torch.nn` or implemented in another module.
- An implementation of `train_epoch()`. If no validation pass is required (eg: for a GAN or other unsupervised model) then `val_epoch()` may be implemented as `pass`.



## Trainer construction
The `Trainer` base class in `lernomatic.train.trainer.py` has the following constructor.


```
class Trainer(object):
    def __init__(self, model=None, **kwargs) -> None:
        self.model           = model
        # Training loop options
        self.num_epochs      :int   = kwargs.pop('num_epochs', 10)
        self.learning_rate   :float = kwargs.pop('learning_rate', 1e-4)
        self.momentum        :float = kwargs.pop('momentum', 0.5)
        self.weight_decay    :float = kwargs.pop('weight_decay', 1e-5)
        self.loss_function   :str   = kwargs.pop('loss_function', 'CrossEntropyLoss')
        self.optim_function  :str   = kwargs.pop('optim_function', 'Adam')
        self.cur_epoch       :int   = 0
        # validation options
        # checkpoint options
        self.checkpoint_dir  :str   = kwargs.pop('checkpoint_dir', 'checkpoint')
        self.checkpoint_name :str   = kwargs.pop('checkpoint_name', 'ck')
        self.save_hist       :bool  = kwargs.pop('save_hist', True)
        # Internal options
        self.verbose         :float = kwargs.pop('verbose', True)
        self.print_every     :int   = kwargs.pop('print_every', 10)
        self.save_every      :float = kwargs.pop('save_every', -1)  # unit is iterations, -1 = save every epoch
        self.save_best       :float = kwargs.pop('save_best', False)
        # Device options
        self.device_id       :int   = kwargs.pop('device_id', -1)
        self.device_map      :float = kwargs.pop('device_map', None)
        # dataset/loader 
        self.batch_size      :int   = kwargs.pop('batch_size', 64)
        self.test_batch_size :int   = kwargs.pop('test_batch_size', 0)
        self.train_dataset          = kwargs.pop('train_dataset', None)
        self.test_dataset           = kwargs.pop('test_dataset', None)
        self.val_dataset            = kwargs.pop('val_dataset', None)
        self.shuffle         :float = kwargs.pop('shuffle', True)
        self.num_workers     :int   = kwargs.pop('num_workers' , 1)
        self.drop_last       :bool  = kwargs.pop('drop_last', True)
        # parameter scheduling
        self.lr_scheduler           = kwargs.pop('lr_scheduler', None)
        self.mtm_scheduler          = kwargs.pop('mtm_scheduler', None)
        self.stop_when_acc   :float = kwargs.pop('stop_when_acc', 0.0)
        self.early_stop      :dict  = kwargs.pop('early_stop', None)

        self.start_epoch = 0
        if self.val_batch_size == 0:
            self.val_batch_size = self.batch_size
        # set up device
        self._init_device()
        # Setup optimizer. If we have no model then assume it will be
        self._init_optimizer()
        # Init the internal dataloader options. If nothing provided assume that
        # we will load options in later (eg: from checkpoint)
        self._init_dataloaders()
        # Init the loss and accuracy history. If no train_loader is provided
        # then we assume that one will be loaded later (eg: in some checkpoint
        # data)
        self._init_history()
        self._send_to_device()

        self.best_acc = 0.0
        if (self.train_loader is not None) and (self.save_every < 0):
            self.save_every = len(self.train_loader)-1
        if self.save_every > 0:
            self.save_best = True

```


## Training and Validation
TODO : explain `train_epoch()` and `val_epoch()`


## Checkpoints
TODO : explain `save_checkpoint()` and `load_checkpoint()`

## History
TODO : explain `save_history()` and `load_history()`


