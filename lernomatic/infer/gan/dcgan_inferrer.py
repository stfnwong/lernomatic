"""
DGCAN_INFERRER
Run the forward pass of a DCGAN model

Stefan Wong 2019
"""

import importlib
import torch
from lernomatic.models import common
from lernomatic.infer import inferrer


class DCGANInferrer(inferrer.Inferrer):
    def __init__(self,
                 model:common.LernomaticModel,
                 **kwargs) -> None:

        self.img_size:int = kwargs.pop('img_size', 64)
        super(DCGANInferrer, self).__init__(model, **kwargs)

    def __repr__(self) -> str:
        return 'DCGANInferrer'

    def generate(self, X:torch.Tensor) -> torch.Tensor:
        self.model.set_eval()
        X = X.to(self.device)
        fake = self.model.forward(X).detach()
        return fake

    def forward(self, X:torch.Tensor=None) -> torch.Tensor:
        if X is None:
            X = torch.randn(1, self.model.get_zvec_dim(), 1, 1)

        fake = self.generate(X)
        return fake.squeeze(0)

    def load_model(self, fname:str) -> None:
        self.model = common.LernomaticModel()
        checkpoint_data = torch.load(fname)

        model_key = 'generator'
        if model_key not in checkpoint_data:
            raise ValueError('No [%s] key in checkpoint data [%s]' % (str(model_key), str(fname)))

        model_params = dict()
        model_params.update({'model_name' : checkpoint_data[model_key]['model_name']})
        model_params.update({'module_name' : checkpoint_data[model_key]['module_name']})
        model_params.update({'model_import_path' : checkpoint_data[model_key]['model_import_path']})
        model_params.update({'module_import_path' : checkpoint_data[model_key]['module_import_path']})
        model_params.update({'model_state_dict' : checkpoint_data[model_key]['model_state_dict']})

        imp = importlib.import_module(checkpoint_data[model_key]['model_import_path'])
        mod = getattr(imp, checkpoint_data[model_key]['model_name'])
        self.model = mod()
        self.model.set_params(model_params)
        self._send_to_device()
