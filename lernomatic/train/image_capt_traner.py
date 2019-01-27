"""
IMAGE_CAPT_TRAINER
Trainer object for image captioning

Stefan Wong 2019
"""

from lernomatic.train import trainer

# NOTE :
class ImageCaptTrainer(trainer.Trainer):
    def __init__(self, encoder, decoder, **kwargs):
        self.verbose = kwargs.pop('verbose', False)

        self.encoder = encoder
        self.decoder = decoder


    def __repr__(self):
        return 'ImageCaptTrainer'
