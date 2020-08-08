"""
TEST_IMAGE_CAPT_TRAINER
Unit tests specific to ImageCaptTrainer object

Stefan Wong 2019
"""

import os
import pytest
import torch
from torchvision import transforms

# units under test
from lernomatic.data.text import word_map
from lernomatic.train.image_caption import image_capt_trainer
from lernomatic.models.image_caption import image_caption
from lernomatic.data.coco import coco_dataset

from test import util


TRAIN_DATASET_PATH = 'hdf5/caption_unit_test_train.h5'
TEST_DATASET_PATH = 'hdf5/caption_unit_test_test.h5'
VAL_DATASET_PATH = 'hdf5/caption_unit_test_val.h5'


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

    if not os.path.exists(TRAIN_DATASET_PATH):
        print('Generating test data...')
        os.system('scripts/coco/gen_caption_unit_test.sh')

    # Create datasets
    train_dataset = coco_dataset.CaptionDataset(
        TRAIN_DATASET_PATH,
        transforms = transforms.Compose([normalize]),
        shuffle=True,
    )
    test_dataset = coco_dataset.CaptionDataset(
        TEST_DATASET_PATH,
        transforms = transforms.Compose([normalize]),
        shuffle=True,
    )
    val_dataset = coco_dataset.CaptionDataset(
        VAL_DATASET_PATH,
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
        device_id     = util.get_device_id(),
        checkpoint_name = 'image_capt_trainer_test',
        batch_size = batch_size,
        # datasets
        train_dataset = train_dataset,
        test_dataset = test_dataset,
        val_dataset = val_dataset
    )

    return trainer


def get_word_map(word_map_path:str = '/mnt/ml-data/datasets/COCO/wordmap.json') -> word_map.WordMap:
    wmap = word_map.WordMap()
    wmap.load(word_map_path)

    return wmap

class TestImageCaptTrainer:
    wmap = get_word_map()
    train_num_epochs = 2
    batch_size = 32

    def test_save_load(self) -> None:
        # Get models
        test_checkpoint = 'checkpoint/image_capt_trainer_test.pkl'
        test_history = 'checkpoint/imag_capt_trainer_test_history.pkl'
        src_encoder, src_decoder = get_model(len(self.wmap))
        assert src_encoder is not None
        assert src_decoder is not None
        src_trainer = get_trainer(
            src_encoder,
            src_decoder,
            self.wmap,
            self.train_num_epochs,
            self.batch_size
        )
        src_trainer.enc_unset_fine_tune()
        assert src_trainer.encoder is not None
        assert src_trainer.decoder is not None
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
            self.batch_size,
        )
        print('Loading checkpoint data from file [%s]' % test_checkpoint)
        dst_trainer.load_checkpoint(test_checkpoint)
        print('Loading history data from file [%s]' % test_history)
        dst_trainer.load_history(test_history)

        # Check the parameters of each model in turn, encoder here
        print('Checking encoders....')
        src_encoder_params = src_trainer.encoder.get_params()
        dst_encoder_params = dst_trainer.encoder.get_params()

        assert len(src_encoder_params) == len(dst_encoder_params)
        for k, v in src_encoder_params.items():
            print('Checking encoder key [%s]' % str(k))
            assert k in dst_encoder_params
            if k == 'model_state_dict':
                continue
            assert v == dst_encoder_params[k]

        src_encoder_net_params = src_trainer.encoder.get_net_state_dict()
        dst_encoder_net_params = dst_trainer.encoder.get_net_state_dict()
        print('Comparing encoder [%s] module parameters...' % repr(dst_trainer.encoder))
        num_pos = 0
        num_neg = 0
        for n, (p1, p2) in enumerate(zip(src_encoder_net_params.items(), dst_encoder_net_params.items())):
            assert p1[0] == p2[0]
            print('Checking %32s [%d/%d] \t' % (str(p1[0]), n+1, len(src_encoder_net_params.items())), end='')
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
        print('Total : %d/%d' % (num_pos, num_pos + num_neg))
        assert num_neg == 0

        # now check the decoder
        print('Checking decoders....')
        src_decoder_params = src_trainer.decoder.get_params()
        dst_decoder_params = dst_trainer.decoder.get_params()

        assert len(src_decoder_params) == len(dst_decoder_params)
        for k, v in src_decoder_params.items():
            assert k in dst_decoder_params
            if k == 'model_state_dict' or k == 'atten_params':
                continue
            assert v == dst_decoder_params[k]

        src_decoder_net_params = src_trainer.decoder.get_net_state_dict()
        dst_decoder_net_params = dst_trainer.decoder.get_net_state_dict()
        print('Comparing decoder [%s] module parameters...' % repr(dst_trainer.decoder))
        num_pos = 0
        num_neg = 0
        for n, (p1, p2) in enumerate(zip(src_decoder_net_params.items(), dst_decoder_net_params.items())):
            assert p1[0] == p2[0]
            print('Checking %32s [%d/%d] \t' % (str(p1[0]), n+1, len(src_decoder_net_params.items())), end='')
            if torch.equal(p1[1], p2[1]):
                num_pos += 1
                print(' [MATCH]')
            else:
                num_neg += 1
                print(' [MISMATCH]')
        print('\n ...done')
        print('%d items correct' % num_pos)
        print('%d items incorrect' % num_neg)
        print('Total : %d/%d' % (num_pos, num_pos + num_neg))
