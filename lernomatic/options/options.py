"""
OPTIONS
Various standard options that can be added to examples, tests, etc

Stefan Wong 2019
"""

import argparse


def get_trainer_options(parser:argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    # Display and save options
    parser.add_argument('--print-every',
                        type=int,
                        default=20,
                        help='Print output every N epochs'
                        )
    parser.add_argument('--save-every',
                        type=int,
                        default=-1,
                        help='Save model checkpoint every N epochs'
                        )
    parser.add_argument('--save-best',
                        action='store_true',
                        default=False,
                        help='Save the best weights in addition to any checkpoints'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of workers to use when generating HDF5 files'
                        )
    # optimization
    parser.add_argument('--loss-function',
                        type=str,
                        default='CrossEntropyLoss',
                        help='Loss function to use. Must be the name of a class in torch.nn (default: CrossEntropyLoss)'
                        )
    parser.add_argument('--optim-function',
                        type=str,
                        default='Adam',
                        help='Optim function to use. Must be the name of a class in torch.optim (default: Adam)'
                        )
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )
    # Training options
    parser.add_argument('--start-epoch',
                        type=int,
                        default=0,
                        help='Epoch to start training from'
                        )
    parser.add_argument('--num-epochs',
                        type=int,
                        default=20,
                        help='Epoch to stop training at'
                        )
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--val-batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use during validation'
                        )
    parser.add_argument('--weight-decay',
                        type=float,
                        default=1e-4,
                        help='Weight decay to use for optimizer'
                        )
    parser.add_argument('--learning-rate',
                        type=float,
                        default=3e-4,
                        help='Learning rate for ADAM optimizer'
                        )
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        help='Momentum for ADAM'
                        )
    parser.add_argument('--grad-clip',
                        type=float,
                        default=5.0,
                        help='Clip gradients at this (absolute) value'
                        )

    return parser


def get_lr_finder_options(parser:argparse.ArgumentParser=None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--find-print-every',
                        type=int,
                        default=64,
                        help='Print output from training loop once every N batches (default: 64)'
                        )
    parser.add_argument('--find-num-epochs',
                        type=int,
                        default=8,
                        help='Number of epochs to train for while searching for parameters'
                        )
    parser.add_argument('--find-lr-min',
                        type=float,
                        default=1e-8,
                        help='Minimum range to begin searching for learning rates (default: 1e-8)'
                        )
    parser.add_argument('--find-lr-max',
                        type=float,
                        default=1.0,
                        help='Maximum range to begin searching for learning rates (default: 1.0)'
                        )
    parser.add_argument('--find-explode-thresh',
                        type=float,
                        default=4.0,
                        help='Stop searching for learning rates when loss is this much larger than minimum loss (default: 4.0)'
                        )
    # TODO : lr_min_factor   and lr_max_scale should be deprecated out in favour of a new edge-based
    # learning rate range selection
    parser.add_argument('--find-lr-select-method',
                        type=str,
                        default='max_acc',
                        help='Method to use when choosing learning rate (default: max_acc)'
                        )


    return parser


    #self.num_epochs       :int   = kwargs.pop('num_epochs', 8)
    ## lr params
    #self.lr_mult          :float = kwargs.pop('lr_mult', 0.0)
    #self.lr_min           :float = kwargs.pop('lr_min', 1e-6)
    #self.lr_max           :float = kwargs.pop('lr_max', 1.0)
    #self.explode_thresh   :float = kwargs.pop('explode_thresh', 4.0)      # fast.ai uses 4 * min_smoothed_loss
    #self.beta             :float = kwargs.pop('beta', 0.999)
    #self.gamma            :float = kwargs.pop('gamma', 0.999995)
    #self.lr_min_factor    :float = kwargs.pop('lr_min_factor', 2.0)
    #self.lr_max_scale     :float = kwargs.pop('lr_max_scale', 1.0)
    #self.lr_select_method :str   = kwargs.pop('lr_select_method', 'max_acc')
    #self.lr_trunc         :int   = kwargs.pop('lr_trunc', 10)
    ## search time
    #self.max_batches      :int   = kwargs.pop('max_batches', 0)
    ## gradient params
    #self.grad_thresh      :float = kwargs.pop('grad_thresh', 0.002)
    ## other
    #self.acc_test         :bool  = kwargs.pop('acc_test', True)
    #self.print_every      :int   = kwargs.pop('print_every', 20)
    #self.verbose          :bool  = kwargs.pop('verbose', False)

    ## can add some unique id (eg: for comparing states from different
    ## experiments)
    #self.expr_id          :str   = kwargs.pop('expr_id', None)
