"""
EX_TRAIN_IMAGE_CAPT
Image captioning training example

Stefan Wong 2019
"""

import argparse
from torchvision import transforms
from lernomatic.train import image_capt_trainer
from lernomatic.models import image_caption
from lernomatic.data.text import word_map
from lernomatic.data.coco import coco_dataset

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

def main():

    if GLOBAL_OPTS['train_data_path'] is None:
        raise ValueError('Must supply a train data path with argument --train-data-path')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )

    wmap = word_map.WordMap()
    wmap.load(GLOBAL_OPTS['wordmap'])

    # Set up encoder and decoder
    decoder = image_caption.DecoderAtten(
        GLOBAL_OPTS['atten_dim'],
        GLOBAL_OPTS['embed_dim'],
        GLOBAL_OPTS['dec_dim'],
        vocab_size = len(wmap),
        #GLOBAL_OPTS['vocab_size'],
        #enc_dim = GLOBAL_OPTS['enc_dim'],
        dropout = GLOBAL_OPTS['dropout'],
        device_id = GLOBAL_OPTS['device_id'],
        verbose = GLOBAL_OPTS['verbose']
    )

    encoder = image_caption.Encoder(
        #GLOBAL_OPTS['feature_size'],
        do_fine_tune = GLOBAL_OPTS['fine_tune']
    )

    # create dataset objects
    train_dataset = coco_dataset.CaptionDataset(
        GLOBAL_OPTS['train_data_path'],
        transforms = transforms.Compose([normalize]),
        shuffle=True,
        num_workers = GLOBAL_OPTS['num_workers'],
        pin_memory  = GLOBAL_OPTS['pin_memory'],
        verbose     = GLOBAL_OPTS['verbose']
    )

    if GLOBAL_OPTS['test_data_path'] is not None:
        test_dataset = coco_dataset.CaptionDataset(
            GLOBAL_OPTS['test_data_path'],
            transforms = transforms.Compose([normalize]),
            shuffle=True,
            num_workers = GLOBAL_OPTS['num_workers'],
            pin_memory  = GLOBAL_OPTS['pin_memory'],
            verbose     = GLOBAL_OPTS['verbose']
        )
    else:
        test_dataset = None


    if GLOBAL_OPTS['val_data_path'] is not None:
        val_dataset = coco_dataset.CaptionDataset(
            GLOBAL_OPTS['val_data_path'],
            transforms = transforms.Compose([normalize]),
            shuffle=True,
            num_workers = GLOBAL_OPTS['num_workers'],
            pin_memory  = GLOBAL_OPTS['pin_memory'],
            verbose     = GLOBAL_OPTS['verbose']
        )
    else:
        val_dataset = None

    if GLOBAL_OPTS['verbose']:
        print('Training dataset :')
        print(train_dataset)
        if test_dataset is not None:
            print('Test dataset :')
            print(test_dataset)
        if val_dataset is not None:
            print('Validation dataset :')
            print(val_dataset)

    # Get a trainer
    trainer = image_capt_trainer.ImageCaptTrainer(
        encoder,
        decoder,
        # training parameters
        batch_size      = GLOBAL_OPTS['batch_size'],
        num_epochs      = GLOBAL_OPTS['num_epochs'],
        learning_rate   = GLOBAL_OPTS['learning_rate'],
        momentum        = GLOBAL_OPTS['momentum'],
        weight_decay    = GLOBAL_OPTS['weight_decay'],
        # word map
        word_map        = wmap,
        # data
        train_dataset   = train_dataset,
        test_dataset    = test_dataset,
        val_dataset     = val_dataset,
        # device
        device_id       = GLOBAL_OPTS['device_id'],
        # checkpoint
        checkpoint_dir  = GLOBAL_OPTS['checkpoint_dir'],
        checkpoint_name = GLOBAL_OPTS['checkpoint_name'],
        # display,
        print_every     = GLOBAL_OPTS['print_every'],
        save_every      = GLOBAL_OPTS['save_every'],
        verbose         = GLOBAL_OPTS['verbose']
    )

    # TODO: try to find a suitable learning rate with param tools

    trainer.train()

def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('--print-every',
                        type=int,
                        default=100,
                        help='Print output every N epochs'
                        )
    parser.add_argument('--save-every',
                        type=int,
                        default=1000,
                        help='Save model checkpoint every N epochs'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of workers to use when generating HDF5 files'
                        )
    parser.add_argument('--pin-memory',
                        default=False,
                        action='store_true',
                        help='If set, pins memory to device'
                        )
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )
    # network options
    parser.add_argument('--embed-dim',
                        type=int,
                        default=512,
                        help='Size of word embedding dim'
                        )
    parser.add_argument('--atten-dim',
                        type=int,
                        default=512,
                        help='Size of attention dim'
                        )
    parser.add_argument('--dec-dim',
                        type=int,
                        default=512,
                        help='Size of decoder dim'
                        )
    parser.add_argument('--dropout',
                        type=float,
                        default=0.5,
                        help='Set dropout rate'
                        )
    # Training options
    parser.add_argument('--enc-lr',
                        type=float,
                        default=1e-4,
                        help='Learning rate for encoder'
                        )
    parser.add_argument('--dec-lr',
                        type=float,
                        default=4e-4,
                        help='Learning rate for decoder'
                        )
    parser.add_argument('--capt-per-img',
                        type=int,
                        default=4,
                        help='Number of captions to use per image'
                        )
    parser.add_argument('--min-word-freq',
                        type=int,
                        default=5,
                        help='Minimum number of times a word should appear in vocab to be included'
                        )
    parser.add_argument('--max-len',
                        type=int,
                        default=100,
                        help='Maximum length allowable for a sentence vector'
                        )
    parser.add_argument('--alpha-c',
                        type=float,
                        default=1.0,
                        help='Regularization parameter for doubly-stochastic attention'
                        )
    parser.add_argument('--grad-clip',
                        type=float,
                        default=5.0,
                        help='Clip gradients at this (absolute) value'
                        )
    parser.add_argument('--fine-tune',
                        action='store_true',
                        default=False,
                        help='Fine tune network'
                        )
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
    parser.add_argument('--weight-decay',
                        type=float,
                        default=1e-4,
                        help='Weight decay to use for optimizer'
                        )
    parser.add_argument('--learning-rate',
                        type=float,
                        default=1e-3,
                        help='Learning rate for optimizer'
                        )
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        help='Momentum for SGD'
                        )
    # Data options
    parser.add_argument('--train-data-path',
                        type=str,
                        default=None,
                        help='Path to train data (HDF5 file)'
                        )
    parser.add_argument('--test-data-path',
                        type=str,
                        default=None,
                        help='Path to test data (HDF5 file)'
                        )
    parser.add_argument('--val-data-path',
                        type=str,
                        default=None,
                        help='Path to validation data (HDF5 file)'
                        )
    # Checkpoint options
    parser.add_argument('--checkpoint-dir',
                        type=str,
                        default='./checkpoint',
                        help='Set directory to place training snapshots into'
                        )
    parser.add_argument('--checkpoint-name',
                        type=str,
                        default='nic_image_capt',
                        help='Name to prepend to all checkpoints'
                        )
    parser.add_argument('--load-checkpoint',
                        type=str,
                        default=None,
                        help='Load a given checkpoint'
                        )
    parser.add_argument('--overwrite',
                        action='store_true',
                        default=False,
                        help='Overwrite existing processed data files'
                        )
    # word map options
    parser.add_argument('--wordmap',
                        type=str,
                        default='wordmap.json',
                        help='Name of wordmap file to load'
                        )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS ---- ')
        for k,v in GLOBAL_OPTS.items():
            print('%s : %s' % (str(k), str(v)))

    main()
