# <a name="trainer-section"></a> TRAINER 

This document details the operation of the `Trainer` module. `Trainer` is both the base class for all trainer objects in `lernomatic` as well as a stand-alone object that can be used to train simple models. 

## <a name="trainer-purpose"></a> Trainer purpose
A `LernomaticModel` is conceptually just a collection of weights and a `forward()` function. While there are typically some utility methods attached to a `LernomaticModel`, the `lernomatic` framework generally consideres the model to just represent the computation graph alone. The specifics of the optimization routine are therefore handled inside the `Trainer` object.

The basic `Trainer` provides enough machinery to perform single-label classification tasks. More specialized trainers can subclass the `Trainer` object and override the `train_epoch()` and `val_epoch()` methods to implement things like Autoencoders, GANs, Seq2Seq models, and so on.


## <a name="train-structure"></a> Trainer structure
At minimum a `Trainer` requires
- A model object
- An optimizer drawn from `torch.optim`
- A loss function drawn from `torch.nn` or implemented in another module.
- An implementation of `train_epoch()`. If no validation pass is required (eg: for a GAN or other unsupervised model) then `val_epoch()` may be implemented as `pass`.

Once the trainer object is [constructed](#train-constructor) the main interface is the `train()` method. Calling this method will cause the trainer to optimize the model with the given parameters

### `train()`
The default `train()` method is a loop that calls `train_epoch()` followed by `val_epoch()` in the range `self.start_epoch` *->* `self.num_epochs`.



## <a name="train-constructor"></a> Trainer construction
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



### <a name="trainer-init"></a> Trainer initialization functions 
The initialization sequence for the base `Trainer` constructor is
- Setup the device for computation by calling `_init_device()`. 


#### `_init_device()`
This function converts an index into a CUDA device. Devices start at index 0 (the first GPU), and increment from there. Passing -1 (or any negative number) sets the device as `torch.device('cpu')`. The index is controlled by the parameter `device_id`, which defaults to -1.

The default implementation of `_init_device()` is given below:

```
    def _init_device(self) -> None:
        if self.device_id < 0:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:%d' % self.device_id)
```

#### `_init_optimizer()`
This module sets up the optimzer for the trainer. Trainers that have multiple optimizers should override this method with the appropriate initialisation for those optimizers.

By default, if `self.model` is `None` then `self.optim` is set to `None`. This is required in case the model is not passed into `__init__()` at construction time (for example, if the model is to be loaded from a checkpoint).

The default implementation of `_init_optimizer()` is given below.

```
    def _init_optimizer(self) -> None:
        if self.model is not None:
            if hasattr(torch.optim, self.optim_function):
                self.optimizer = getattr(torch.optim, self.optim_function)(
                    self.model.get_model_parameters(),
                    lr = self.learning_rate,
                    weight_decay = self.weight_decay
                )
            else:
                raise ValueError('Cannot find optim function %s' % str(self.optim_function))
        else:
            self.optimizer = None

        # Get a loss function
        if hasattr(nn, self.loss_function):
            loss_obj = getattr(nn, self.loss_function)
            self.criterion = loss_obj()
        else:
            raise ValueError('Cannot find loss function [%s]' % str(self.loss_function))


```


#### `_init_dataloaders()`
The `Trainer` constructor accepts dataset objects. These should be of type `torch.util.data.Dataset` or a subclass of this type. If these are not `None`, then a dataloader is created for each of the `train_dataset`, `'test_dataset`, and `val_dataset` arguments respectively. These are used in the corresponding `_epoch()` methods (so `train_dataloader` is used in `train_epoch()`, `val_dataloader` is used in `val_epoch()`, so on).


The default implementation of `_init_dataloaders()` is given below:

```
    def _init_dataloaders(self) -> None:
        if self.train_dataset is None:
            self.train_loader = None
        else:
            self.train_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size = self.batch_size,
                drop_last = self.drop_last,
                shuffle = self.shuffle
            )

        if self.test_dataset is None:
            self.test_loader = None
        else:
            self.test_loader = torch.utils.data.Dataloader(
                self.test_dataset,
                batch_size = self.val_batch_size,
                drop_last = self.drop_last,
                shuffle    = self.shuffle
            )

        if self.val_dataset is None:
            self.val_loader = None
        else:
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size = self.val_batch_size,
                drop_last = self.drop_last,
                shuffle    = False
            )
```


#### `_init_history()`
The training history is maintained in a seperate `numpy.ndarray`. The default configuration provides arrays for the loss history, test loss history, and accuracy history. More specific implementations should overrride both the `_init_history()` method, and also update the `save_history()` and `load_history()` methods to correctly save and load the changes to disk during training.

The default implementation of `_init_history()` is given below:

```
    def _init_history(self) -> None:
        self.loss_iter      = 0
        self.val_loss_iter  = 0
        self.acc_iter       = 0
        self.iter_per_epoch = int(len(self.train_loader) / self.num_epochs)

        if self.train_loader is not None:
            self.loss_history   = np.zeros(len(self.train_loader) * self.num_epochs)
        else:
            self.loss_history = None

        if self.val_loader is not None:
            self.val_loss_history = np.zeros(len(self.val_loader) * self.num_epochs)
            self.acc_history = np.zeros(len(self.val_loader) * self.num_epochs)
        else:
            self.val_loss_history = None
            self.acc_history = None
```


#### `_send_to_device()`
Finally the model(s) are sent the the target device. In the simple case, this is just a wrapper around `LernomaticModel.send_to()` with the trainers device as the argument. More complex trainers may contain multiple models, and in those cases this method should be overridden to send all required components to the correct device (or devices, should that be required).

The default implementation of `_send_to_device()` is given below:

```
    def _send_to_device(self) -> None:
        self.model.send_to(self.device)
``` 


## <a name="trainer-train-val"></a> Training and Validation
TODO : explain `train_epoch()` and `val_epoch()`


## <a name="trainer-checkpoints"></a> Checkpoints
TODO : explain `save_checkpoint()` and `load_checkpoint()`

## <a name="trainer-histrory"></a> History
TODO : explain `save_history()` and `load_history()`


