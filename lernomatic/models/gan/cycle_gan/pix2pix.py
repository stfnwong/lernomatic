"""
PIX2PIX
Model that implements Pix2Pix

Stefan Wong 2019
"""

from lernomatic.models import common


class Pix2Pix(common.LernomatcModel):
    def __init__(self, **kwargs) -> None:
        self.net               : torch.nn.Module = None
        self.import_path       : str             = 'lernomatic.model.common'
        self.model_name        : str             = 'LernomaticModel'
        self.module_name       : str             = None
        self.module_import_path: str             = None
