"""
GAN_IMAGE_PROC
Make HDF5 datasets of images for training GANs

Stefan Wong 2019
"""
import os
from PIL import Image
import argparse
#from tqdm import tqdm
from lernomatic.data import data_split
from lernomatic.data import image_proc

GLOBAL_OPTS = dict()


def main() -> None:

    if GLOBAL_OPTS['outfile'] is None:
        raise ValueError('No outfile specified (use --outfile=OUTFILE)')

    img_paths = os.listdir(GLOBAL_OPTS['dataset_root'])
    src_len = len(img_paths)
    if img_paths is None or len(img_paths) < 1:
        raise ValueError('No files in directory [%s]' % GLOBAL_OPTS['dataset_root'])

    if GLOBAL_OPTS['verbose']:
        print('Found %d files in path [%s]' % (src_len, str(GLOBAL_OPTS['dataset_root'])))

    print('Checking files in path [%s]' % GLOBAL_OPTS['dataset_root'])
    num_err = 0
    for n, path in enumerate(img_paths):
        if GLOBAL_OPTS['verbose']:
            print('Checking file [%d / %d] ' % (n+1, len(img_paths)), end='\r')

        # If there are any exceptions then just remove the file that caused
        # them and continue
        try:
            img = Image.open(GLOBAL_OPTS['dataset_root'] + str(path)).convert('RGB')
        except:
            img_paths.pop(n)
            num_err += 1
        if img is None:
            if GLOBAL_OPTS['verbose']:
                print('Failed to load image [%d/%d] <%s>' % (n, len(img_paths), str(path)))
            img_paths.pop(n)
            num_err += 1

    # get a split
    s = data_split.DataSplit(split_name=GLOBAL_OPTS['split_name'])
    s.data_paths  = [GLOBAL_OPTS['dataset_root'] + str(path) for path in img_paths]
    s.data_labels = [0 for x in range(len(img_paths))]
    s.elem_ids    = [0 for x in range(len(img_paths))]

    # process the data
    proc = image_proc.ImageDataProc(
        label_dataset_size = 1,
        image_dataset_size = (3, GLOBAL_OPTS['image_size'], GLOBAL_OPTS['image_size']),
        to_tensor = True,
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
    # data options
    parser.add_argument('--image-size',
                        type=int,
                        default=128,
                        help='Resize all images to be this size (squared) (default: 128)'
                        )
    # split options
    parser.add_argument('--split-name',
                        type=str,
                        default='GANDATA',
                        help='Name for data split (default: GANDATA)'
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
