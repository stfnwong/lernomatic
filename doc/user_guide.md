# LERNOMATIC USER GUIDE 


## Basic components of LERNOMATIC
The two major components in LERNOMATIC are *models* and *trainers*. 

### Models
A model is a wrapper around a Pytorch nn.Module. It contains an nn.Module (which in turn contains the weights for the network), as well as meta information about its components. 

The basic model in `lernomatic.model.common` has the following constructor

```
class LernomaticModel(object):
    def __init__(self, **kwargs) -> None:
        self.net               : torch.nn.Module = None
        self.import_path       : str             = 'lernomatic.model.common'
        self.model_name        : str             = 'LernomaticModel'
        self.module_name       : str             = None
        self.module_import_path: str             = None

```

### Trainers
Trainers take a model, datasets and training parameters and perform training on the model. A trainer contains.

#### Constructor
The basic trainer model in `lernomatic.train.trainer` has the following constructor

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
        # parameter scheduling
        self.lr_scheduler           = kwargs.pop('lr_scheduler', None)
        self.mtm_scheduler          = kwargs.pop('mtm_scheduler', None)
        self.stop_when_acc   :float = kwargs.pop('stop_when_acc', 0.0)
        self.early_stop      :dict  = kwargs.pop('early_stop', None)

        if self.test_batch_size == 0:
            self.test_batch_size = self.batch_size
        self.best_acc = 0.0
        if self.save_every > 0:
            self.save_best = True

        # Setup optimizer. If we have no model then assume it will be
        self._init_optimizer()
        # set up device
        self._init_device()
        # Init the internal dataloader options. If nothing provided assume that
        # we will load options in later (eg: from checkpoint)
        self._init_dataloaders()
        # Init the loss and accuracy history. If no train_loader is provided
        # then we assume that one will be loaded later (eg: in some checkpoint
        # data)
        if self.train_loader is not None:
            self._init_history()

        self._send_to_device()

```

Each trainer is expected to provide a `train()` method that trains the model. In the default implementation, sub routines are implemented for training and testing/validation. It is recommended that this approach be followed for trainers, however the only API requirement is that a `train()` method be exposed.

#### Training parameters
TODO : talk about how the parameters function


### Creating a new model
A model contains two parts. 

1. An `LernomaticModel` wrapper. 
2. One or more `torch.nn.Module' components.

The base model class is the `LernomaticModel` which is located in `lernomatic.model.common`. All models in LERNOMATIC should inherit from this class. 

As an example, consider the following basic Pytorch network [taken from the pytorch tutorials](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html).

```
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

```

To create an LernomaticModel object that can be used with the Lernomatic Trainer class, we need to wrap this `nn.Module` object in an `LernomaticModule` object. For the sake of example, let the above network be placed into a file called 'new_network.py' located in the LERNOMATIC models module `lernomatic.model`.

```
 #in this example, the LernomaticModel object is also is new_network.py
import torch
from lernomatic.model import common


class LernomaticModelNet(common.LernomaticModel):
    def __init__(self) -> None:
        self.net = Net()        # instantiate the actual class here 
        # we provide the location for this file  in the import_path attribute
        self.import_path = 'lernomatic.model.new_network.py'       
        # we provide the path for the Net() class in the module_import_path attribute
        self.module_import_path = 'lernomatic.model.new_network.py'
        # we also provide the name of this object
        self.model_name = 'LernomaticModelNet'
        # and the name of the Net() module
        self.module_name = 'Net'


    # we create a hook to the forward() method of self.net
    def forward(self, X:torch.Tensor) -> torch.Tensor:
        return self.net.forward(X)
```

By default, the `LernomaticModel` class expects there to be a single `torch.nn.Module` object at the attribute `net`. As such, the basic `LernomaticModel` operations are written with this in mind. In cases where there are two or more networks, consider combining them into a single `torch.nn.Module`. If this is not feasible or appropriate, you may need to provide overrides for some of the default getters or setters in order to allow the trainer object to save and load the model checkpoints correctly. 


### Creating a new Trainer 

To create a new trainer, we inherit from the trainer object in `lernomatic.train.trainer`.


```
from lernomatic.train import trainer

class NewTrainer(trainer.Trainer):
    def __init__(self, model, **kwargs) -> None:
        """
        Init codes goes here, (eg: get keyword args)
        """

```

