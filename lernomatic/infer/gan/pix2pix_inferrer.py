"""
PIX2PIX_INFERRER
Wraps forward-pass operation for pix2pix models

Stefan Wong 2019
"""

import torch
import importlib
from lernomatic.models import common
from lernomatic.infer import inferrer


class Pix2PixInferrer(inferrer.Inferrer):
    """
    Pix2PixInferrer

    Wraps the forward pass of a model. This is the counterpart to
    the Trainer module. This does the forward pass, handles devices,
    and so on
    """
    def __init__(self, model:common.LernomaticModel=None, **kwargs) -> None:
        super(Pix2PixInferrer, self).__init__(model, **kwargs)

    def __repr__(self) -> str:
        return 'Pix2PixInferrer'

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        self.model.set_eval()
        return self.model.forward(X).detach()

    def load_model(self, fname:str, model_key:str='g_net')-> None:
        self.model = common.LernomaticModel()
        checkpoint_data = torch.load(fname)

        if model_key not in checkpoint_data:
            raise ValueError('No model key [%s] in checkpoint file %s' % (str(model_key), str(fname)))

        if 'gen_params' not in checkpoint_data[model_key]:
            raise ValueError('Expecting [gen_params] key in checkpoint data [%s]' % str(fname))

        model_params = dict()
        model_params.update({'model_name'         : checkpoint_data[model_key]['model_name']})
        model_params.update({'module_name'        : checkpoint_data[model_key]['module_name']})
        model_params.update({'model_import_path'  : checkpoint_data[model_key]['model_import_path']})
        model_params.update({'module_import_path' : checkpoint_data[model_key]['module_import_path']})
        model_params.update({'model_state_dict'   : checkpoint_data[model_key]['model_state_dict']})
        model_params.update({'gen_params'         : checkpoint_data[model_key]['gen_params']})

        imp = importlib.import_module(checkpoint_data[model_key]['model_import_path'])
        mod = getattr(imp, checkpoint_data[model_key]['model_name'])

        if 'num_downsamples' in model_params['gen_params']:
            self.model = mod(
                num_input_channels = checkpoint_data[model_key]['gen_params']['num_input_channels'],
                num_output_channels = checkpoint_data[model_key]['gen_params']['num_output_channels'],
                num_downsamples = checkpoint_data[model_key]['gen_params']['num_downsamples'],
            )
        else:
            self.model = mod(
                num_input_channels = checkpoint_data[model_key]['gen_params']['num_input_channels'],
                num_output_channels = checkpoint_data[model_key]['gen_params']['num_output_channels'],
            )
        self.model.set_params(model_params)
        self._send_to_device()

    def get_input_dims(self) -> tuple:
        pass
