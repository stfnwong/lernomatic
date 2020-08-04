"""
GAN_IMAGE_PROC
Make HDF5 datasets of images for training GANs

Stefan Wong 2019
"""
import os
from PIL import Image
import argparse
from tqdm import tqdm
#from tqdm import tqdm
from lernomatic.data import data_split
from lernomatic.data import image_proc
from lernomatic.util import file_util

# debug
#

GLOBAL_OPTS = dict()


def main() -> None:
    if GLOBAL_OPTS['outfile'] is None:
        raise ValueError('No outfile specified (use --outfile=OUTFILE)')

    image_paths = file_util.get_file_paths(
        GLOBAL_OPTS['dataset_root'],
        verbose=GLOBAL_OPTS['verbose']
    )

    if len(image_paths) == 0:
        raise ValueError('Failed to find any images in path or subpaths of [%s]' % GLOBAL_OPTS['dataset_root'])

    print('Checking %d files starting at path [%s]' % (len(image_paths), GLOBAL_OPTS['dataset_root']))
    num_err = 0
    for n, path in enumerate(tqdm(image_paths, unit='images')):
        # If there are any exceptions then just remove the file that caused
        # them and continue
        try:
            img = Image.open(path)
        except:
            image_paths.pop(n)
            num_err += 1
            continue

        if img is None:
            if GLOBAL_OPTS['verbose']:
                print('Failed to load image [%d/%d] <%s>' % (n, len(image_paths), str(path)))
            image_paths.pop(n)
            num_err += 1

        if (GLOBAL_OPTS['size'] > 0) and ((n - num_err) >= GLOBAL_OPTS['size']):
            print('\nFound %d files, stopping checks' % GLOBAL_OPTS['size'])
            break

    if GLOBAL_OPTS['randomize']:
        print('Randomizing...')
        image_paths = random.shuffle(image_paths)

    # get a split
    s = data_split.DataSplit(split_name=GLOBAL_OPTS['split_name'])

    if GLOBAL_OPTS['size'] > 0:
        s.data_paths  = [str(path) for path in image_paths[0 : GLOBAL_OPTS['size']]]
        s.data_labels = [int(0) for _ in range(GLOBAL_OPTS['size'])]
        s.elem_ids    = [int(0) for _ in range(GLOBAL_OPTS['size'])]
    else:
        s.data_paths  = [str(path) for path in image_paths]
        s.data_labels = [int(0) for _ in range(len(image_paths))]
        s.elem_ids    = [int(0) for _ in range(len(image_paths))]
    s.has_labels  = True
    s.has_ids     = True

    # process the data
    proc = image_proc.ImageDataProc(
        label_dataset_size = 1,
        image_dataset_size = (3, GLOBAL_OPTS['image_size'], GLOBAL_OPTS['image_size']),
        to_tensor = True,
        to_pil = GLOBAL_OPTS['to_pil'],
        verbose = GLOBAL_OPTS['verbose']
    )
    proc.proc(s, GLOBAL_OPTS['outfile'])


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root',
                        type=str,
                        help='Path to root of data'
                        )
    parser.add_argument('--outfile',
                        type=str,
                        default=None,
                        help='Filename for output HDF5 file'
                        )
    parser.add_argument('--size',
                        type=int,
                        default=0,
                        help='Size of dataset. If 0, use all data points found (default: 0)'
                        )
    # data options
    parser.add_argument('--image-size',
                        type=int,
                        default=128,
                        help='Resize all images to be this size (squared) (default: 128)'
                        )
    parser.add_argument('--extension',
                        type=str,
                        default='jpg',
                        help='Select the file extension to search for in dataset (default: jpg)'
                        )
    parser.add_argument('--randomize',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    # split options
    parser.add_argument('--split-name',
                        type=str,
                        default='GANDATA',
                        help='Name for data split (default: GANDATA)'
                        )
    parser.add_argument('--to-pil',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )

    return parser



if __name__ == '__main__':
    parser = arg_parser()
    args = vars(parser.parse_args())

    for k, v in args.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose']:
        print('---- GLOBAL OPTIONS ----')
        for k, v in GLOBAL_OPTS.items():
            print('\t [%s] : %s' % (str(k), str(v)))

    main()
