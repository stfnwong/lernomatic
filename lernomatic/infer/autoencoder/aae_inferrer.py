"""
AAE_INFERRER
Run forward (generator) pass on AAE model

Stefan Wong 2019
"""

import torch
from lernomatic.models import common
from lernomatic.infer import inferrer

# debug
#from pudb import set_trace; set_trace()


class AAEInferrer(inferrer.Inferrer):
    def __init__(self,
                 q_net:common.LernomaticModel=None,
                 p_net:common.LernomaticModel=None,
                 **kwargs) -> None:
        self.q_net = q_net
        self.p_net = p_net

        super(AAEInferrer, self).__init__(None, **kwargs)

    def __repr__(self) -> str:
        return 'AAEInferrer'

    def _send_to_device(self):
        if self.q_net is not None:
            self.q_net.send_to(self.device)

        if self.p_net is not None:
            self.p_net.send_to(self.device)

    def load_model(self, fname: str) -> None:
        """
        Load model data from checkpoint
        """
        checkpoint_data = torch.load(fname)
        self.set_trainer_params(checkpoint_data['trainer_params'])

        # Load the models
        # P-Net
        model_import_path = checkpoint_data['p_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['p_net']['model_name'])
        self.p_net = mod()
        self.p_net.set_params(checkpoint_data['p_net'])
        # Q-Net
        model_import_path = checkpoint_data['q_net']['model_import_path']
        imp = importlib.import_module(model_import_path)
        mod = getattr(imp, checkpoint_data['q_net']['model_name'])
        self.q_net = mod()
        self.q_net.set_params(checkpoint_data['q_net'])

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        X.resize_(X.shape[0], self.q_net.get_x_dim())

        self.q_net.set_eval()
        self.p_net.set_eval()
        q_out = self.q_net.forward(X)
        if type(q_out) is tuple:
            z_in = torch.cat((q_out[0], q_out[1]), 1)
            z = self.p_net.forward(z_in)
        else:
            z = self.p_net.forward(q_out)

        return z



class AAEUnsupervisedInferrer(AAEInferrer):
    def __init__(self,
                 q_net:common.LernomaticModel=None,
                 p_net:common.LernomaticModel=None,
                 **kwargs) -> None:
        super(AAEUnsupervisedInferrer, self).__init__(q_net, p_net, **kwargs)

    def __repr__(self) -> str:
        return 'AAEUnsupervisedInferrer'

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        X.resize_(X.shape[0], self.q_net.get_x_dim())

        self.q_net.set_eval()
        self.p_net.set_eval()
        q_out = self.q_net.forward(X)
        z = self.p_net.forward(q_out)

        return z



class AAESemiInferrer(AAEInferrer):
    def __init__(self,
                 q_net:common.LernomaticModel=None,
                 p_net:common.LernomaticModel=None,
                 **kwargs) -> None:
        super(AAESemiInferrer, self).__init__(q_net, p_net, **kwargs)

    def __repr__(self) -> str:
        return 'AAESemiInferrer'

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        X = X.to(self.device)
        X.resize_(X.shape[0], self.q_net.get_x_dim())

        self.q_net.set_eval()
        self.p_net.set_eval()
        q_out = self.q_net.forward(X)
        z_in = torch.cat((q_out[0], q_out[1]), 1)
        z = self.p_net.forward(z_in)

        return z
