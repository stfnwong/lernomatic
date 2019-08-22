# MODEL


# Constructor
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

The purpose of the `LernomaticModel` wrapper is largerly to allow models to maintain their own metadata. This means that models can be largely self loading, and can therefore be attached to any `Trainer` object without needing to keep extra state.


