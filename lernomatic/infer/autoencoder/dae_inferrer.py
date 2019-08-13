"""
DAE_INFERRER
Forward pass wrapper for De-noising Autoencoder models

Stefan Wong 2019
"""

import importlib
import torch
from lernomatic.models import common
from lernomatic.infer import inferrer


class DAEInferrer(inferrer.Inferrer):
    def __init__(self,
                 encoder:common.LernomaticModel=None,
                 decoder:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.encoder:common.LernomaticModel = encoder
        self.decoder:common.LernomaticModel = decoder
        # noise options
        self.noise_bias:float   = kwargs.pop('noise_bias', 0.25)
        self.noise_factor:float = kwargs.pop('noise_factor', 0.1)

        super(DAEInferrer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'DAEInferrer'

    def _send_to_device(self) -> None:
        if self.encoder is not None:
            self.encoder.send_to(self.device)
        if self.decoder is not None:
            self.decoder.send_to(self.device)

    def get_noise(self, X:torch.Tensor) -> torch.Tensor:
        noise = torch.rand(*X.shape)
        return torch.mul(X + self.noise_bias, self.noise_factor * noise)

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        # corrupt input with noise
        X_noise = self.get_noise(X)
        output = self.encoder.forward(X_noise)
        output = self.decoder.forward(output)

        return output.detach()

    def load_model(self, fname:str) -> None:
        """
        Load model data from checkpoint
        """
        checkpoint_data = torch.load(fname)

        # Load the models
        # Encoder
        model_import_path = checkpoint_data['encoder']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['encoder']['model_name'])
        self.encoder = mod()
        self.encoder.set_params(checkpoint_data['encoder'])

        # Decoder
        model_import_path = checkpoint_data['decoder']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['decoder']['model_name'])
        self.decoder = mod()
        self.decoder.set_params(checkpoint_data['decoder'])
