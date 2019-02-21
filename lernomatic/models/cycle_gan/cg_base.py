"""
CG_BASE
Basic CycleGAN model

Stefan Wong 2019
"""

class CycleGANBase(object):
    """
    Base class for CycleGAN models
    """
    def __init__(self, **kwargs):
        valid_init_types = ('xavier', 'kaiming', 'normal', 'orthogonal')
        self.num_input_channels  = kwargs.pop('num_input_channels', 3)  # 3 for RGB, 1 for gray
        self.num_output_channels = kwargs.pop('num_ouput_channels', 3)  # 3 for RGB, 1 for gray
        self.num_gen_filters     = kwargs.pop('num_gen_filters', 64)
        self.num_dis_filters     = kwargs.pop('num_dis_filters', 64)
        self.use_batchnorm       = kwargs.pop('use_batchnorm', True)
        self.dropout             = kwargs.pop('dropout', 0.0)
        self.init_type           = kwargs.pop('init_type', 'xavier')

        # TODO : source also has options to specify architecture of D and G

        if self.init_type not in valid_init_types:
            raise ValueError('Unknown init type [%s], must be one of %s' %\
                             (str(self.init_type), str(valid_init_types))
            )

    def __repr__(self):
        return 'CycleGANBase'
