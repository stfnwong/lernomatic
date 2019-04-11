"""
COCO_HDF5_INSPECTOR
Look inside a COCO HDF5 file

Stefan Wong 2019
"""

import sys
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from lernomatic.util import hdf5_util
from lernomatic.data.text import word_map

# debug
#from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()
GLOBAL_TOOL_MODES = ('dump-elem','dump-capt', 'check-shape')
GLOBAL_WORD_MAP = None


def dump_caption(elem_idx) -> None:
    hdf5_data = hdf5_util.HDF5Data(
        GLOBAL_OPTS['input'],
        verbose=GLOBAL_OPTS['verbose']
    )

    caption = hdf5_data.get_elem(
        GLOBAL_OPTS['caption_name'],
        elem_idx,
    )
    caplen = hdf5_data.get_elem(
        GLOBAL_OPTS['caplen_name'],
        elem_idx
    )
    if isinstance(caplen, np.ndarray) or isinstance(caplen, list):
        caplen = caplen[0]

    if GLOBAL_WORD_MAP is not None:
        caption_string = []
        for w in caption:
            caption_string.append(GLOBAL_WORD_MAP.lookup_word(w))

    if GLOBAL_WORD_MAP is not None:
        print('Caption <%d> : [%s]' % (elem_idx, caption_string[0 : caplen]))
    else:
        print('Caption <%d> : [%s]' % (elem_idx, caption[0 : caplen]))
    print('Caplen : %d' % caplen)


def dump_element(elem_idx) -> None:
    hdf5_data = hdf5_util.HDF5Data(
        GLOBAL_OPTS['input'],
        verbose=GLOBAL_OPTS['verbose']
    )

    img = hdf5_data.get_elem(
        GLOBAL_OPTS['feature_name'],
        elem_idx
    )
    caption = hdf5_data.get_elem(
        GLOBAL_OPTS['caption_name'],
        elem_idx
    )
    caplen = hdf5_data.get_elem(
        GLOBAL_OPTS['caplen_name'],
        elem_idx
    )
    if isinstance(caplen, np.ndarray) or isinstance(caplen, list):
        caplen = caplen[0]

    # Transpose image dimensions
    img = img.transpose(2,1,0)
    if GLOBAL_WORD_MAP is not None:
        caption_string = []
        for w in caption:
            caption_string.append(GLOBAL_WORD_MAP.lookup_word(w))


    # Actually on reflection this seems a bit complicated...
    if GLOBAL_OPTS['verbose']:
        print('Image dimensions : %s' % str(img.shape))
        if GLOBAL_WORD_MAP is not None:
            print('Caption : [%s]' % caption_string[0 : caplen])
        else:
            print('Caption : [%s]' % caption[0 : caplen])
        print('Caplen  :  %d'  % caplen)

    # generate plot
    if GLOBAL_WORD_MAP is not None:
        title = 'Image <%d> caption: [%s] ' %\
            (elem_idx, caption_string[0 : caplen])
    else:
        title = 'Image <%d> caption: [%s] ' %\
            (elem_idx, str(caption[0 : caplen]))
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    fig.tight_layout()

    if GLOBAL_OPTS['output'] is not None:
        fig.savefig(GLOBAL_OPTS['output'])
    else:
        plt.imshow(fig)
        plt.show()


def check_shape() -> None:

    hdf5_data = hdf5_util.HDF5Data(
        GLOBAL_OPTS['input'],
        verbose=GLOBAL_OPTS['verbose']
    )

    if not hdf5_data.has_dataset(GLOBAL_OPTS['key']):
        raise ValueError('No dataset [%s] in file [%s]' %\
                         (str(GLOBAL_OPTS['key']), str(GLOBAL_OPTS['input']))
        )

    pos_match = 0
    neg_match = 0
    neg_match_idxs = []
    for idx in tqdm(range(hdf5_data.get_size(GLOBAL_OPTS['key'])), unit='index'):
        #print('Checking element [%d / %d]' % (idx, hdf5_data.get_size(GLOBAL_OPTS['key'])), end='\r')
        elem_data = hdf5_data.get_elem(GLOBAL_OPTS['key'], idx)
        if elem_data.shape == GLOBAL_OPTS['shape']:
            pos_match += 1
        else:
            neg_match += 1
            neg_match_idxs.append(idx)


    print('\n DONE')
    print('Required shape %s' % str(GLOBAL_OPTS['shape']))
    print('%d / %d element(s) match' % (pos_match, hdf5_data.get_size(GLOBAL_OPTS['key'])))
    print('%d / %d element(s) do not match' % (neg_match, hdf5_data.get_size(GLOBAL_OPTS['key'])))
    if len(neg_match_idxs) > 0:
        print('Index(es) of elements with incorrect shape:')
        print(neg_match_idxs)


def arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        help='Input file'
                        )
    parser.add_argument('--mode',
                        type=str,
                        default='dump-elem',
                        help='Tool mode. Select from %s (default: dump-elem)' % str(GLOBAL_TOOL_MODES)
                        )
    parser.add_argument('--wordmap',
                        type=str,
                        default=None,
                        help='Wordmap file. If present captions are converted into words (default: None)'
                        )
    parser.add_argument('--output',
                        type=str,
                        default=None,
                        help='File to place output figure in. If unspecified figure is displayed (default: None)'
                        )
    parser.add_argument('--vis-index',
                        type=int,
                        default=0,
                        help='Index of dataset to visualize'
                        )
    parser.add_argument('--vis-range',
                        type=str,
                        default=None,
                        help='A comma seperated range of indicies to use'
                        )
    parser.add_argument('--feature-name',
                        type=str,
                        default='images',
                        help='Name of dataset containing data to visualize'
                        )
    parser.add_argument('--caption-name',
                        type=str,
                        default='captions',
                        help='Name of dataset containing caption information'
                        )
    parser.add_argument('--caplen-name',
                        type=str,
                        default='caplens',
                        help='Name of dataset containing caption length information'
                        )
    # Key argument
    parser.add_argument('--key',
                        type=str,
                        default=None,
                        help='Dataset key. (default: None)'
                        )
    # shape argument
    parser.add_argument('--shape',
                        type=int,
                        nargs='+',
                        default=None,
                        help='Required shape to check. Format should be space-seperated list of sizes (eg: --shape 3 256 256) (default: None)'
                        )
    # general args
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

    if GLOBAL_OPTS['input'] is None:
        print('ERROR: no input file specified.')
        sys.exit(1)

    vis_start, vis_end, vis_step = 0, 0, 0
    if GLOBAL_OPTS['vis_range'] is not None:
        vis_range = GLOBAL_OPTS['vis_range'].split(',')
        if len(vis_range) == 2:
            vis_start = int(vis_range[0])
            vis_end   = int(vis_range[1])
            vis_step  = 1
        elif len(vis_range) == 3:
            vis_start = int(vis_range[0])
            vis_end   = int(vis_range[1])
            vis_step  = int(vis_range[2])
        else:
            raise ValueError('Invalid range option [%s], should be start,end or start,end,step' %\
                             (str(GLOBAL_OPTS['vis_range']))
            )

    # load the wordmap, if we have one
    if GLOBAL_OPTS['wordmap'] is not None:
        GLOBAL_WORD_MAP = word_map.WordMap()
        GLOBAL_WORD_MAP.load(GLOBAL_OPTS['wordmap'])

    if GLOBAL_OPTS['mode'] == 'dump-elem':
        if GLOBAL_OPTS['vis_range'] is not None:
            for idx in range(vis_start, vis_end, vis_step):
                dump_element(idx)
        else:
            dump_element(GLOBAL_OPTS['vis_index'])
    elif GLOBAL_OPTS['mode'] == 'dump-capt':
        if GLOBAL_OPTS['vis_range'] is not None:
            for idx in range(vis_start, vis_end, vis_step):
                dump_caption(idx)
        else:
            dump_caption(GLOBAL_OPTS['vis_index'])
    elif GLOBAL_OPTS['mode'] == 'check-shape':
        if GLOBAL_OPTS['key'] is None:
            raise RuntimeError('mode [check-shape] requires dataset key (use --key argument)')
        if GLOBAL_OPTS['shape'] is None:
            raise RuntimeError('mode [check-shape] requires shape (use --shape argument)')
        GLOBAL_OPTS['shape'] = tuple(GLOBAL_OPTS['shape'])
        check_shape()
    else:
        raise RuntimeError('Invalid tool mode [%s]' % str(GLOBAL_OPTS['mode']))
