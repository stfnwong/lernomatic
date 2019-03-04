"""
COCO_HDF5_INSPECTOR
Look inside a COCO HDF5 file

Stefan Wong 2019
"""

import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from lernomatic.util import hdf5_util
from lernomatic.data.text import word_map

# debug
from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()
GLOBAL_TOOL_MODES = ('dump-elem','dump-capt')
GLOBAL_WORD_MAP = None


def dump_caption():
    hdf5_data = hdf5_util.HDF5Data(
        GLOBAL_OPTS['input'],
        verbose=GLOBAL_OPTS['verbose']
    )

    caption = hdf5_data.get_elem(
        GLOBAL_OPTS['caption_name'],
        GLOBAL_OPTS['vis_index']
    )
    caplen = hdf5_data.get_elem(
        GLOBAL_OPTS['caplen_name'],
        GLOBAL_OPTS['vis_index']
    )
    if isinstance(caplen, np.ndarray) or isinstance(caplen, list):
        caplen = caplen[0]

    if GLOBAL_WORD_MAP is not None:
        caption_string = []
        for w in caption:
            caption_string.append(GLOBAL_WORD_MAP.lookup_word(w))

    if GLOBAL_WORD_MAP is not None:
        print('Caption <%d> : [%s]' % (GLOBAL_OPTS['vis_index'], caption_string[0 : caplen]))
    else:
        print('Caption <%d> : [%s]' % (GLOBAL_OPTS['vis_index'], caption[0 : caplen]))
    print('Caplen : %d' % caplen)


def dump_element(elem_idx):
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



def arg_parser():
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
            vis_start = vis_range[0]
            vis_end   = vis_range[1]
            vis_step  = 0
        elif len(vis_range) == 3:
            vis_start = vis_range[0]
            vis_end   = vis_range[1]
            vis_step  = vis_range[2]
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
        dump_caption()
    else:
        raise RuntimeError('Invalid tool mode [%s]' % str(GLOBAL_OPTS['mode']))
