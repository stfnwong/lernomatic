"""
EX_TRAIN_IMAGE_CAPT
Image captioning training example

Stefan Wong 2019
"""

import sys
import argparse
import torch
from torchvision import transforms
from lernomatic.param import image_caption_lr
from lernomatic.train import schedule
from lernomatic.train import image_capt_trainer
from lernomatic.models import image_caption
from lernomatic.models import common
from lernomatic.data.text import word_map
from lernomatic.data.coco import coco_dataset

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def get_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )

    return transforms.Compose([normalize])

# get a finder object
def get_lr_finder(trainer:image_capt_trainer.ImageCaptTrainer,
                  find_type:str='CaptionLogFinder',
                  **kwargs) -> image_caption_lr.CaptionLogFinder:

    if not hasattr(image_caption_lr, find_type):
        raise ValueError('Unknown learning rate finder type [%s]' % str(find_type))

    # get keyword args
    lr_select_method = kwargs.pop('lr_select_method', 'max_acc')
    num_epochs       = kwargs.pop('num_epochs', 4)
    explode_thresh   = kwargs.pop('explode_thresh', 6.0)
    print_every      = kwargs.pop('print_every', 100)

    lr_find_obj = getattr(image_caption_lr, find_type)
    lr_finder = lr_find_obj(
        trainer,
        lr_min           = 1e-7,
        lr_max           = 1.0,
        lr_select_method = lr_select_method,
        num_epochs       = num_epochs,
        explode_thresh   = explode_thresh,
        print_every      = print_every
    )

    return lr_finder


# get a scheduler object
def get_scheduler(lr_min:float,
                  lr_max:float,
                  sched_type:str='TriangularScheduler',
                  stepsize:int=6000) -> schedule.LRScheduler:
    if sched_type is None:
        return None

    if not hasattr(schedule, sched_type):
        raise ValueError('Unknown scheduler type [%s]' % str(sched_type))

    lr_sched_obj = getattr(schedule, sched_type)
    lr_scheduler = lr_sched_obj(
        stepsize = stepsize,
        lr_min = lr_min,
        lr_max = lr_max
    )

    return lr_scheduler


def get_word_map(fname:str) -> word_map.WordMap:
    wmap = word_map.WordMap()
    wmap.load(fname)
    print('Loaded word map containing %d words' % len(wmap))

    return wmap


def get_models(wmap:word_map.WordMap) -> tuple:
    encoder = image_caption.Encoder(
        do_fine_tune = GLOBAL_OPTS['fine_tune']
    )

    decoder = image_caption.DecoderAtten(
        atten_dim  = GLOBAL_OPTS['atten_dim'],
        embed_dim  = GLOBAL_OPTS['embed_dim'],
        dec_dim    = GLOBAL_OPTS['dec_dim'],
        vocab_size = len(wmap),
        dropout    = GLOBAL_OPTS['dropout'],
        verbose    = GLOBAL_OPTS['verbose']
    )

    return (encoder, decoder)


def get_trainer(encoder:common.LernomaticModel,
                decoder:common.LernomaticModel,
                wmap:word_map.WordMap,
                train_dataset:coco_dataset.CaptionDataset=None,
                test_dataset: coco_dataset.CaptionDataset=None,
                val_dataset:  coco_dataset.CaptionDataset=None
                ) -> image_capt_trainer.ImageCaptTrainer:
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
        early_stop      = {'num_epochs' : 20, 'improv': 0.05},
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

    return trainer


def get_dataset(fname:str,
                transforms,
                num_workers:int,
                pin_memory:bool,
                verbose:bool=False,
                shuffle:bool=False
                ) -> coco_dataset.CaptionDataset:

    # TODO : type hint for transform
    dataset = coco_dataset.CaptionDataset(
        fname,
        transforms = transforms,
        shuffle=shuffle,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    return dataset


def overfit() -> None:
    # Try to overfit the classifier on some small amount of data

    overfit_dataset = get_dataset(
        GLOBAL_OPTS['overfit_train_data'],
        get_transforms(),
        GLOBAL_OPTS['num_workers'],
        GLOBAL_OPTS['pin_memory'],
        verbose = GLOBAL_OPTS['verbose'],
        shuffle=True
    )

    if GLOBAL_OPTS['overfit_test_data'] is not None:
        overfit_test_data = get_dataset(
            GLOBAL_OPTS['overfit_test_data'],
            get_transforms(),
            GLOBAL_OPTS['num_workers'],
            GLOBAL_OPTS['pin_memory'],
            verbose = GLOBAL_OPTS['verbose'],
            shuffle=True
        )
    else:
        overfit_test_data = None

    wmap = get_word_map(GLOBAL_OPTS['wordmap'])
    encoder, decoder = get_models(wmap)
    trainer = get_trainer(
        encoder,
        decoder,
        wmap,
        train_dataset = overfit_dataset,
        test_dataset = overfit_test_data
    )
    trainer.checkpoint_name = 'nic_overfit_test'
    trainer.save_every = 0
    trainer.enc_set_fine_tune()
    trainer.train()


# ======== MAIN ======== #
def main() -> None:

    if GLOBAL_OPTS['train_data_path'] is None:
        raise ValueError('Must supply a train data path with argument --train-data-path')

    wmap = get_word_map(GLOBAL_OPTS['wordmap'])
    encoder, decoder = get_models(wmap)

    # create dataset objects
    train_dataset = get_dataset(
        GLOBAL_OPTS['train_data_path'],
        get_transforms(),
        GLOBAL_OPTS['num_workers'],
        GLOBAL_OPTS['pin_memory'],
        verbose = GLOBAL_OPTS['verbose'],
        shuffle = True
    )

    if GLOBAL_OPTS['test_data_path'] is not None:
        test_dataset = get_dataset(
            GLOBAL_OPTS['test_data_path'],
            get_transforms(),
            GLOBAL_OPTS['num_workers'],
            GLOBAL_OPTS['pin_memory'],
            verbose     = GLOBAL_OPTS['verbose'],
            shuffle=True,
        )
    else:
        test_dataset = None

    if GLOBAL_OPTS['val_data_path'] is not None:
        val_dataset = get_dataset(
            GLOBAL_OPTS['val_data_path'],
            get_transforms(),
            GLOBAL_OPTS['num_workers'],
            GLOBAL_OPTS['pin_memory'],
            verbose     = GLOBAL_OPTS['verbose'],
            shuffle=True,
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

    if GLOBAL_OPTS['lr_find_test_path'] is not None:
        lr_find_dataset = get_dataset(
            GLOBAL_OPTS['lr_find_test_path'],
            get_transforms(),
            GLOBAL_OPTS['num_workers'],
            GLOBAL_OPTS['pin_memory'],
            verbose     = GLOBAL_OPTS['verbose'],
            shuffle=True,
        )
    else:
        lr_find_dataset = None

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
        early_stop      = {'num_epochs' : 2, 'improv': 0.05},
        grad_clip       = GLOBAL_OPTS['grad_clip'],
        dec_lr          = GLOBAL_OPTS['dec_lr'],
        enc_lr          = GLOBAL_OPTS['enc_lr'],
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
    # TODO : I wonder if its better to just pre-train with a fixed schedule
    # and then run the learning rate finder on the resulting trained network
    # get lr finder
    if GLOBAL_OPTS['find_lr'] is True:
        lr_find_train_dataset = get_dataset(
            GLOBAL_OPTS['overfit_train_data'],
            None,
            1,
            False
        )
        lr_find_val_dataset  = get_dataset(
            GLOBAL_OPTS['overfit_val_data'],
            None,
            1,
            False
        )

        # get actual LRFinder object
        lr_finder = get_lr_finder(trainer)
        lr_finder.set_train_dataloader(torch.utils.data.DataLoader(lr_find_train_dataset))
        lr_finder.set_val_dataloader(torch.utils.data.DataLoader(lr_find_val_dataset))
        lr_min, lr_max = lr_finder.find()
        print('Found LR range as %.4f -> %.4f' % (lr_min, lr_max))
        scheduler = get_scheduler(lr_min, lr_max, GLOBAL_OPTS['sched_type'], stepsize = len(trainer.train_dataset) / 2)
        if GLOBAL_OPTS['verbose']:
            print('Created scheduler [%s]\n %s' % (repr(scheduler), str(scheduler)))
        trainer.set_lr_scheduler(scheduler)
    else:
        lr_finder = None
        enc_scheduler = get_scheduler(1e-6, GLOBAL_OPTS['enc_lr'], 'DecayWhenAcc')
        dec_scheduler = get_scheduler(1e-6, GLOBAL_OPTS['dec_lr'], 'DecayWhenAcc')
        if GLOBAL_OPTS['verbose']:
            print('Created scheduler (decoder) [%s]\n %s' % (repr(enc_scheduler), str(enc_scheduler)))
            print('Created scheduler (encoder) [%s]\n %s' % (repr(dec_scheduler), str(dec_scheduler)))
        trainer.set_enc_lr_scheduler(enc_scheduler)
        trainer.set_dec_lr_scheduler(dec_scheduler)

    # If we load a checkpoint, we need to decide what to do for the 'rest' of
    # the training schedule

    # Training schedule
    print('Training for %d epochs without fine tuning' % trainer.get_num_epochs())
    # First train for 30 epochs
    trainer.enc_unset_fine_tune()
    trainer.train()

    # Turn on fine tuning and train for another 30 epochs
    trainer.enc_set_fine_tune()
    trainer.set_batch_size(GLOBAL_OPTS['fine_tune_batch_size'])
    trainer.set_num_epochs(60)
    print('Added fine tuning, training until %d epochs with batch size %d' %\
      (trainer.get_num_epochs(), trainer.get_batch_size())
    )
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
                        default=50,
                        help='Print output every N epochs'
                        )
    parser.add_argument('--save-every',
                        type=int,
                        default=-1,
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
    parser.add_argument('--find-lr',
                        action='store_true',
                        default=False,
                        help='Use the learing rate finder to select a learning rate'
                        )

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
    parser.add_argument('--fine-tune-batch-size',
                        type=int,
                        default=32,
                        help='Batch size to use during training'
                        )
    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.0,
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
    # scheduling options
    parser.add_argument('--sched-type',
                        type=str,
                        default='DecayWhenAcc',
                        help='Scheduler to use during training (default: None)'
                        )
    # finder options
    parser.add_argument('--lr-select-method',
                        type=str,
                        default='max_acc',
                        help='Heuristic to use when selecting lr ranges (default: max_acc)'
                        )
    parser.add_argument('--find-num-epochs',
                        type=int,
                        default=8,
                        help='Maximum number of epochs to attempt to find learning rate'
                        )
    parser.add_argument('--find-explode-thresh',
                        type=float,
                        default=4.5,
                        help='Threshold at which to stop increasing learning rate'
                        )
    parser.add_argument('--exp-decay',
                        type=float,
                        default=0.001,
                        help='Exponential decay term'
                        )
    parser.add_argument('--sched-stepsize',
                        type=int,
                        default=4000,
                        help='Size of step for learning rate scheduler'
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
    parser.add_argument('--lr-find-test-path',
                        type=str,
                        default=None,
                        help='Path to test data for lr finder (HDF5 file)'
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
    # test options
    parser.add_argument('--overfit',
                        action='store_true',
                        default=False,
                        help='Perform overfit test then exit'
                        )
    parser.add_argument('--overfit-train-data',
                        type=str,
                        default=None,
                        help='Path to train dataset to overfit on'
                        )
    parser.add_argument('--overfit-val-data',
                        type=str,
                        default=None,
                        help='Path to val dataset to overfit on'
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
            print('\t[%s] : %s' % (str(k), str(v)))

    if GLOBAL_OPTS['overfit'] is True:
        overfit()
        sys.exit(1)

    main()
