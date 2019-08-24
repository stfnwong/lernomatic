# INFERRER
The `Inferrer` object is the counterpart to the [`Trainer`](#trainer-section) object. It is intended to wrap the forward pass for a model.



## <a name="inferrer-constructor"></a> Inferrer Constructor

The default constructor for the base `Inferrer` class in `lernomatic.infer.inferrer.py` is as follows


```python

    def __init__(self, model=None, **kwargs) -> None:
        self.model = model
        self.device_id:int = kwargs.pop('device_id', -1)

        self._init_device()
        self._send_to_device()

```

The `_init_device()` and `_send_to_device()` methods here work the same way as the methods in the [`Trainer`](#trainer-init) do.


### `forward()` method 
The default `forward()` method simply wraps the `forward()` call for the corresponding `LernomaticModel`. The default implementation is shown below.

```python

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        return self.model.forward(X)
```

For more complex models, additional steps that may be required at inference time can be implemented here. For example the `DAEInferrer` module in `lernomatic.infer.autoencoder.dae_inferrer.py` implements the forward pass for a Denoising-Autoencoder. The forward method for that class is defined as 

```python

    def get_noise(self, X:torch.Tensor) -> torch.Tensor:
        noise = torch.rand(*X.shape)
        return torch.mul(X + self.noise_bias, self.noise_factor * noise)

```

This object has some controls for the amount of noise that is added to the input tensor `X`, and performs the noise addition along with the forward pass.


### `load_model()`
This method is an analogue to the `load_checkpoint()` method in the [`Trainer` object](#trainer-checkpoints). Models can be loaded directly from Trainer checkpoints.

There are cases when this method should be specialized. For example, if loading model that has encoding and decoding steps, where the steps are contained in two seperate `LernomaticModel` objects.
