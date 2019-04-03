"""
EX_TRAIN_IMAGE_CAPT
Image captioning training example

Stefan Wong 2019
"""

import argparse
from torchvision import transforms
from lernomatic.param import image_caption_lr
from lernomatic.train import schedule
from lernomatic.train import image_capt_trainer
from lernomatic.models import image_caption
from lernomatic.data.text import word_map
from lernomatic.data.coco import coco_dataset

# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()

# TODO : make argument local to this function (ie: take out all GLOBAL_OPTS
# references)
def get_lr_finder(trainer, find_type='CaptionLogFinder'):
    if not hasattr(image_caption_lr, find_type):
        raise ValueError('Unknown learning rate finder type [%s]' % str(find_type))

    lr_find_obj = getattr(image_caption_lr, find_type)
    lr_finder = lr_find_obj(
        trainer,
        lr_min         = 1e-7,
        lr_max         = 1.0,
        lr_select_method = GLOBAL_OPTS['lr_select_method'],
        num_epochs     = GLOBAL_OPTS['find_num_epochs'],
        explode_thresh = GLOBAL_OPTS['find_explode_thresh'],
        print_every    = GLOBAL_OPTS['print_every']
    )

    return lr_finder


def get_scheduler(lr_min, lr_max, sched_type='TriangularScheduler'):
    if sched_type is None:
        return None

    if not hasattr(schedule, sched_type):
        raise ValueError('Unknown scheduler type [%s]' % str(sched_type))

    lr_sched_obj = getattr(schedule, sched_type)
    lr_scheduler = lr_sched_obj(
        stepsize = GLOBAL_OPTS['sched_stepsize'],    # TODO : optimal stepsize selection?
        lr_min = lr_min,
        lr_max = lr_max
    )

    return lr_scheduler


# ======== MAIN ======== #
def main() -> None:

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
    # get lr finder
    if GLOBAL_OPTS['find_lr'] is True:
        lr_finder = get_lr_finder(trainer)
        lr_min, lr_max = lr_finder.find()
        print('Found LR range as %.4f -> %.4f' % (lr_min, lr_max))
        scheduler = get_scheduler(lr_min, lr_max, GLOBAL_OPTS['sched_type'])
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

    # Training schedule
    print('Training for %d epochs without fine tunining' % trainer.get_num_epochs())
    # First train for 30 epochs
    trainer.enc_unset_fine_tune()
    trainer.train()

    # Turn on fine tuning and train for another 30 epochs
    # TODO : the batch size will need to be reduced here
    trainer.enc_set_fine_tune()
    trainer.set_batch_size(GLOBAL_OPTS['fine_tune_batch_size'])
    trainer.set_num_epochs(60)
    print('Added fine tuning, training until %d epochs' % trainer.get_num_epochs())
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
                        default=64,
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
                        default=None,
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
