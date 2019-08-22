# MODEL

# Model Components
A `LernomaticModel` is just a simple wrapper around a `torch.nn.Module`. The wrapper mainly holds metadata about the class name and module location so that models can be self loading into [Trainers](trainer.md) or [Inferrers](inferrer.md)

# <a name="model-create-new"></a> Creating a new Lernomatic Model 
To create an LernomaticModel object that can be used with the Lernomatic Trainer class, we need to wrap this `nn.Module` object in an `LernomaticModule` object. For the sake of example, let the above network be placed into a file called `new_network.py` located in the LERNOMATIC models module `lernomatic.model`.

```python

# model definition new_network.py

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


# Constructor
The basic model in `lernomatic.model.common` has the following constructor

```python

class LernomaticModel(object):
    def __init__(self, **kwargs) -> None:
        self.net               : torch.nn.Module = None
        self.import_path       : str             = 'lernomatic.model.common'
        self.model_name        : str             = 'LernomaticModel'
        self.module_name       : str             = None
        self.module_import_path: str             = None

```

The purpose of the `LernomaticModel` wrapper is largerly to allow models to maintain their own metadata. This means that models can be largely self loading, and can therefore be attached to any `Trainer` object without needing to keep extra state.


