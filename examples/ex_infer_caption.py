"""
EX_INFER_CAPTION
Do forward pass on caption model

Stefan Wong 2019
"""

import argparse
from torchvision import transforms
import torch
import cv2
import numpy as np
# lernomatic modules
from lernomatic.data.text import word_map
from lernomatic.infer import infer_caption

GLOBAL_OPTS = dict()

def main() -> None:

    # Read in input files
    img = cv2.imread(GLOBAL_OPTS['input'])
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)

    img = cv2.resize(
        img,
        (GLOBAL_OPTS['img_size'], GLOBAL_OPTS['img_size']),
        interpolation=cv2.INTER_CUBIC
    )
    img = img.transpose(2, 0, 1)
    img = img / 255.0
    img = torch.FloatTensor(img)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_transform = transforms.Compose([normalize])
    image = img_transform(img)

    wmap = word_map.WordMap()
    wmap.load(GLOBAL_OPTS['wordmap'])
    infer = infer_caption.CaptionInferrer(
        wmap,
        beam_size = GLOBAL_OPTS['beam_size'],
        max_steps = 50
    )

    infer.load_checkpoint(GLOBAL_OPTS['checkpoint'])
    caption = infer.gen_caption(image)
    print(str(caption))


def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('checkpoint',
                        type=str,
                        help='Path to checkpoint file'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
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
    # Input options
    parser.add_argument('--input',
                        type=str,
                        default=None,
                        help='Path to input image'
                        )
    parser.add_argument('--img-size',
                        type=int,
                        default=256,
                        help='Resize images to be a square of this size (default: 256)'
                        )
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )
    # word map options
    parser.add_argument('--wordmap',
                        type=str,
                        default='wordmap.json',
                        help='Name of wordmap file to load'
                        )
    # caption generation options
    parser.add_argument('--beam-size',
                        type=int,
                        default=3,
                        help='Beam size to use during beam search (default: 3)'
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
