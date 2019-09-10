# LERNOMATIC USER GUIDE 

LERNOMATIC is one of a dime-a-dozen PyTorch wrappers that people have surely written in the past few years. Like all the others, the real heavy lifting is being done by the PyTorch libraries (ATEN, torch, caffe2, and so on). The library implements various types of models, including 

- Resnets
- Variational Autoencoders
- Deep Convolutional GANs

and others (this file to be updated as more models are completed). The rest of this guide outlines the basic structure of the library and gives a brief overview of its idioms and how to extend it.


## Basic components of LERNOMATIC
The two major components in LERNOMATIC are *models* and *trainers*. 

### Models
A model is a wrapper around a Pytorch nn.Module. It contains an nn.Module (which in turn contains the weights for the network), as well as meta information about its components. 



### Trainers
Trainers take a model, datasets and training parameters and perform training on the model. Trainers also contain all the training and validation logic, and keep records of the training and validation history


Each trainer is expected to provide a `train()` method that trains the model. In the default implementation, sub routines are implemented for training and testing/validation. It is recommended that this approach be followed for trainers, however the only API requirement is that a `train()` method be exposed.

By default, the `train()` method calls `train_epoch()` in a loop that ranges from `self.start_epoch` to `self.num_epochs`. The actual training logic is implemented in `train_epoch()`. 

If a validation loader is present, then `val_epoch()` is called as well. This implements the validation logic on a single epoch of the validation data. The order of calls is always `train_epoch()` followed by `val_epoch()`. In other words, a complete epoch of training is done followed by one complete epoch of validation (they are not interleaved).

For a complete description of the `Trainer` object, see [the trainer section](trainer.md).


### Creating a new model
A model contains two parts. 

1. A `LernomaticModel` wrapper. 
2. One or more `torch.nn.Module` components.

The base model class is the `LernomaticModel` which is located in `lernomatic.model.common`. All models in LERNOMATIC should inherit from this class. 

As an example, consider the following basic Pytorch network [taken from the pytorch tutorials](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html).

```python

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

For a complete description of models in *LERNOMATIC* [see the model section](models.md)



### Creating a new Trainer 

To create a new trainer, we inherit from the trainer object in `lernomatic.train.trainer`.


```python

from lernomatic.train import trainer

class NewTrainer(trainer.Trainer):
    def __init__(self, model, **kwargs) -> None:
        """
        Init codes goes here, (eg: get keyword args)
        """

```

The `train_epoch()` method is where the training logic should go. 

