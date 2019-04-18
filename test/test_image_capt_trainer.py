"""
TEST_IMAGE_CAPT_TRAINER
Unit tests specific to ImageCaptTrainer object

Stefan Wong 2019
"""

import os
import sys
import argparse
import unittest
import torch
from torchvision import transforms

# units under test
from lernomatic.data.text import word_map
from lernomatic.train import image_capt_trainer
from lernomatic.models import image_caption
from lernomatic.data.coco import coco_dataset

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def get_model(vocab_size) -> tuple:

    encoder = image_caption.Encoder()
    decoder = image_caption.DecoderAtten(
        atten_dim  = 512,
        embed_dim  = 512,
        dec_dim    = 512,
        vocab_size = vocab_size
    )

    return (encoder, decoder)


def get_trainer(encoder:image_caption.Encoder,
                decoder:image_caption.DecoderAtten,
                wmap:word_map.WordMap,
                num_epochs:int,
                batch_size:int) -> image_capt_trainer.ImageCaptTrainer:

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )

    # Create datasets
    train_dataset = coco_dataset.CaptionDataset(
        GLOBAL_OPTS['train_dataset_path'],
        transforms = transforms.Compose([normalize]),
        shuffle=True,
    )
    test_dataset = coco_dataset.CaptionDataset(
        GLOBAL_OPTS['test_dataset_path'],
        transforms = transforms.Compose([normalize]),
        shuffle=True,
    )
    val_dataset = coco_dataset.CaptionDataset(
        GLOBAL_OPTS['val_dataset_path'],
        transforms = transforms.Compose([normalize]),
        shuffle=True,
    )

    # Create trainer modules
    trainer = image_capt_trainer.ImageCaptTrainer(
        encoder,
        decoder,
        num_epochs = num_epochs,
        word_map = wmap,
        print_every = 10,
        save_every = 0,
        device_id = GLOBAL_OPTS['device_id'],
        checkpoint_name = 'image_capt_trainer_test',
        batch_size = batch_size,
        # datasets
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        val_dataset = val_dataset
    )

    return trainer


class TestImageCaptTrainer(unittest.TestCase):

    def setUp(self):
        self.wmap = word_map.WordMap()
        self.wmap.load(GLOBAL_OPTS['wordmap'])
        self.train_num_epochs = 2

    def test_save_load(self):
        print('======== TestImageCaptTrainer.test_save_load ')

        # Get models
        test_checkpoint = 'checkpoint/image_capt_trainer_test.pkl'
        test_history = 'checkpoint/imag_capt_trainer_test_history.pkl'
        src_encoder, src_decoder = get_model(len(self.wmap))
        src_trainer = get_trainer(
            src_encoder,
            src_decoder,
            self.wmap,
            self.train_num_epochs,
            GLOBAL_OPTS['batch_size']
        )
        print('Created new trainer object')
        src_trainer.train()
        src_trainer.save_checkpoint(test_checkpoint)
        src_trainer.save_history(test_history)

        # Load eveything into a new trainer
        dst_trainer = get_trainer(
            None,
            None,
            self.wmap,
            self.train_num_epochs,
            GLOBAL_OPTS['batch_size']
        )
        print('Loading checkpoint data from file [%s]' % test_checkpoint)
        dst_trainer.load_checkpoint(test_checkpoint)
        print('Loading history data from file [%s]' % test_history)
        dst_trainer.load_history(test_history)

        # Check the parameters of each model in turn, encoder here
        print('Checking encoders....')
        src_encoder_params = src_trainer.encoder.get_params()
        dst_encoder_params = dst_trainer.encoder.get_params()

        self.assertEqual(len(src_encoder_params), len(dst_encoder_params))
        for k, v in src_encoder_params.items():
            print('Checking encoder key [%s]' % str(k))
            self.assertIn(k, dst_encoder_params.keys())
            if k == 'model_state_dict':
                continue
            self.assertEqual(v, dst_encoder_params[k])

        src_encoder_net_params = src_trainer.encoder.get_net_state_dict()
        dst_encoder_net_params = dst_trainer.encoder.get_net_state_dict()
        print('Comparing encoder module parameters...')
        num_pos = 0
        num_neg = 0
        for n, (p1, p2) in enumerate(zip(src_encoder_net_params.items(), dst_encoder_net_params.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t' % (str(p1[0]), n+1, len(src_encoder_net_params.items())), end='')
            if torch.equal(p1[1], p2[1]):
                num_pos += 1
                print(' [MATCH]')
            else:
                num_neg += 1
                print(' [MISMATCH]')
            #self.assertEqual(True, torch.equal(p1[1], p2[1]))
        print('\n ...done')
        print('%d items correct' % num_pos)
        print('%d items incorrect' % num_neg)

        # now check the decoder
        print('Checking decoders....')
        src_decoder_params = src_trainer.decoder.get_params()
        dst_decoder_params = dst_trainer.decoder.get_params()

        self.assertEqual(len(src_decoder_params), len(dst_decoder_params))
        for k, v in src_decoder_params.items():
            self.assertIn(k, dst_decoder_params.keys())
            if k == 'model_state_dict' or k == 'atten_params':
                continue
            self.assertEqual(v, dst_decoder_params[k])

        src_decoder_net_params = src_trainer.decoder.get_net_state_dict()
        dst_decoder_net_params = dst_trainer.decoder.get_net_state_dict()
        print('Comparing decoder module parameters...')
        num_pos = 0
        num_neg = 0
        for n, (p1, p2) in enumerate(zip(src_decoder_net_params.items(), dst_decoder_net_params.items())):
            self.assertEqual(p1[0], p2[0])
            print('Checking parameter %s [%d/%d] \t' % (str(p1[0]), n+1, len(src_decoder_net_params.items())), end='')
            if torch.equal(p1[1], p2[1]):
                num_pos += 1
                print(' [MATCH]')
            else:
                num_neg += 1
                print(' [MISMATCH]')
            #self.assertEqual(True, torch.equal(p1[1], p2[1]))
        print('\n ...done')
        print('%d items correct' % num_pos)
        print('%d items incorrect' % num_neg)

        print('======== TestImageCaptTrainer.test_save_load <END>')


# Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
                        action='store_true',
                        default=False,
                        help='Sets verbose mode'
                        )
    parser.add_argument('--draw-plot',
                        action='store_true',
                        default=False,
                        help='Draw plots'
                        )
    parser.add_argument('--num-workers',
                        type=int,
                        default=1,
                        help='Number of worker processes to use for reading HDF5'
                        )
    parser.add_argument('--device-id',
                        type=int,
                        default=0,
                        help='Device to use for tests (default : -1)'
                        )
    # wordmap
    parser.add_argument('--wordmap',
                        type=str,
                        default='/mnt/ml-data/datasets/COCO/wordmap.json',
                        help='Wordmap file to use for test'
                        )
    # dataset files (run tools/gen_caption_unit_test.sh to generate)
    parser.add_argument('--batch-size',
                        type=int,
                        default=32,
                        help='Batch size to use during test'
                        )
    parser.add_argument('--train-dataset-path',
                        type=str,
                        default='hdf5/caption_unit_test_train.h5',
                        help='Path to unit test train dataset'
                        )
    parser.add_argument('--test-dataset-path',
                        type=str,
                        default='hdf5/caption_unit_test_test.h5',
                        help='Path to unit test test dataset'
                        )
    parser.add_argument('--val-dataset-path',
                        type=str,
                        default='hdf5/caption_unit_test_val.h5',
                        help='Path to unit test val dataset'
                        )

    # args for unittest module
    parser.add_argument('unittest_args', nargs='*')

    args = parser.parse_args()
    arg_vals = vars(args)
    for k, v in arg_vals.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print('-------- GLOBAL OPTS (%s) --------' % str(sys.argv[0]))
        for k, v in GLOBAL_OPTS.items():
            print('\t[%s] : %s' % (str(k), str(v)))

    # Check that the unit test data is present
    if not os.path.exists(GLOBAL_OPTS['train_dataset_path']):
        raise RuntimeError('No training data - run tools/gen_caption_unit_test.sh to generate')
    if not os.path.exists(GLOBAL_OPTS['test_dataset_path']):
        raise RuntimeError('No testing data - run tools/gen_caption_unit_test.sh to generate')
    if not os.path.exists(GLOBAL_OPTS['val_dataset_path']):
        raise RuntimeError('No validation data - run tools/gen_caption_unit_test.sh to generate')

    sys.argv[1:] = args.unittest_args
    unittest.main()


