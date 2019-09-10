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
        self.import_path        : str             = 'lernomatic.model.common'
        self.module_import_path : str             = None
        self.model_name         : str             = 'LernomaticModel'
        self.module_name        : str             = None

```

The purpose of the `LernomaticModel` wrapper is largerly to allow models to maintain their own metadata. This means that models can be largely self loading, and can therefore be attached to any `Trainer` object without needing to keep extra state. The following sections outline the purpose of each parameter.

#### `import_path`
This represents the path to the `LernomaticModule` object itself. 

#### `module_import_path`
This represents the path to the `torch.nn.Module` object that the `LernomaticModel` is wrapping.



# `forward()` method
This wraps the `forward()` or `__call__()` method of a `torch.nn.Module` object. The use of `__call__()` in the `LernomaticModel` itself is not actually implemented at the time of writing.


# Self loading complex models.
In some cases, just knowing the paths and names of the modules will not be enough to have a model self load. This can happen when the module constructor dynamically generates the network structure. Often this requires that the size of the final network be passed to the constructor. In other cases, the default keyword arguments may not align with those in the checkpoint. In these cases, some additional steps may be required to support self-loading.

The `get_params()` and `set_params()` methods are used to save and load additional model specific parameters. For example, the DCGAN Generator in `lernomatic.models.gan.dcgan.py` has the following definition.


```python

class DCGANGenerator(common.LernomaticModel):
    def __init__(self, **kwargs) -> None:
        self.net = DCGANGeneratorModule(**kwargs)
        self.model_name         = 'DCGANGenerator'
        self.module_name        = 'DCGANGeneratorModule'
        self.import_path        = 'lernomatic.models.gan.dcgan'
        self.module_import_path = 'lernomatic.models.gan.dcgan'

        self.init_weights()

    def __repr__(self) -> str:
        return 'DCGGenerator'

    def get_zvec_dim(self) -> int:
        return self.net.zvec_dim

    def get_num_filters(self) -> int:
        return self.net.num_filters

    def zero_grad(self) -> None:
        self.net.zero_grad()

    def get_num_blocks(self) -> int:
        return self.net.num_blocks

    def get_block_filter_sizes(self) -> list:
        return self.net.block_filter_sizes

    def init_weights(self) -> None:
        classname = self.net.__class__.__name__
        if classname.find('Conv') != -1:
            self.net.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            self.net.weight.data.normal_(1.0, 0.02)
            self.net.bias.data.fill_(0)

    def get_params(self) -> dict:
        params = super(DCGANGenerator, self).get_params()
        params['gen_params'] = {
            'num_filters'  : self.net.num_filters,
            'num_channels' : self.net.num_channels,
            'kernel_size'  : self.net.kernel_size,
            'img_size'     : self.net.img_size,
            'zvec_dim'     : self.net.zvec_dim
        }

        return params

    def set_params(self, params : dict) -> None:
        # regular model stuff
        self.import_path = params['model_import_path']
        self.model_name  = params['model_name']
        self.module_name = params['module_name']
        self.module_import_path = params['module_import_path']
        # Import the actual network module
        imp = importlib.import_module(self.module_import_path)
        mod = getattr(imp, self.module_name)
        self.net = mod(
            num_channels = params['gen_params']['num_channels'],
            num_filters  = params['gen_params']['num_filters'],
            kernel_size  = params['gen_params']['kernel_size'],
            img_size     = params['gen_params']['img_size'],
            zvec_dim     = params['gen_params']['zvec_dim'],
        )
        self.net.load_state_dict(params['model_state_dict'])

```

Here the additional model specific parameters `num_filters`, `num_channels`, `kernel_size`, `img_size`, and `zvec_dim` are saved into a new dictionary which is saved along with the `nn.Module`. To restore the model at load time we call `set_params()`, which imports the module and calls the constructor with the arguments provided by `get_params()`.
