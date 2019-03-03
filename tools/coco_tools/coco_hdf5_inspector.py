"""
COCO_HDF5_INSPECTOR
Look inside a COCO HDF5 file

Stefan Wong 2019
"""

import sys
import argparse
from matplotlib import pyplot as plt
from lernomatic.util import hdf5_util

# debug
from pudb import set_trace; set_trace()


GLOBAL_OPTS = dict()


def main():

    # TODO: add support for comma seperated list arguments
    hdf5_data = hdf5_util.HDF5Data(
        GLOBAL_OPTS['input'],
        verbose=GLOBAL_OPTS['verbose']
    )

    data_params = dict()

    data_params['img'] = hdf5_data.get_elem(
        GLOBAL_OPTS['feature_name'],
        GLOBAL_OPTS['vis_index']
    )
    data_params['caption'] = hdf5_data.get_elem(
        GLOBAL_OPTS['caption_name'],
        GLOBAL_OPTS['vis_index']
    )
    data_params['caplen'] = hdf5_data.get_elem(
        GLOBAL_OPTS['caplen_name'],
        GLOBAL_OPTS['vis_index']
    )

    # Transpose image dimensions
    data_params['img'] = data_params['img'].transpose(2,1,0)

    # Actually on reflection this seems a bit complicated...
    if GLOBAL_OPTS['verbose']:
        for k, v in data_params.items():
            if k == 'img':
                print('\t[%s] : %s' % (str(k), str(v.shape)))
            else:
                print('\t[%s] : %s' % (str(k), str(v)))

    # generate plot
    #title = 'Image <%d> caplen <%s> caption: [%s] ' % (GLOBAL_OPTS['vis_index'], str(caplen), caption)
    title = 'Image <%d> caption: [%s] ' %\
        (GLOBAL_OPTS['vis_index'], data_params['caption'][-1 : data_params['caplen']])
    fig, ax = plt.subplots()
    ax.imshow(data_params['img'])
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
                        default='captions',
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

    main()

