# LERNOMATIC USER GUIDE 


## Basic components of LERNOMATIC
The two major components in LERNOMATIC are *models* and *trainers*. 

### Models
A model is a wrapper around a Pytorch nn.Module. It contains an nn.Module (which in turn contains the weights for the network), as well as meta information about its components. 



### Trainers
Trainers take a model, datasets and training parameters and perform training on the model. Trainers also contain all the training and validation logic, and keep records of the training and validation history


Each trainer is expected to provide a `train()` method that trains the model. In the default implementation, sub routines are implemented for training and testing/validation. It is recommended that this approach be followed for trainers, however the only API requirement is that a `train()` method be exposed.

By default, the `train()` method calls `train_epoch()` in a loop that ranges from `self.start_epoch` to `self.num_epochs`. The actual training logic is implemented in `train_epoch()`. 

If a validation loader is present, then `val_epoch()` is called as well. This implements the validation logic on a single epoch of the validation data. The order of calls is always `train_epoch()` followed by `val_epoch()`. In other words, a complete epoch of training is done followed by one complete epoch of validation (they are not interleaved).

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

The `train_epoch()` method is where the training logic should go. 

