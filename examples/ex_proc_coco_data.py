"""
EX_PROC_COCO_DATA
Example for processing COCO dataset for use with classifier, caption
generator, etc

Stefan Wong 2019
"""

import sys
import argparse
# data split object
#from lernomatic.data.coco import data
from lernomatic.data.coco import coco_data
from lernomatic.data.coco import word_map
from lernomatic.data.coco import coco_proc

# debug
#from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def main():

    train_data = coco_data.COCODataSplit(
        GLOBAL_OPTS['coco_json'],
        GLOBAL_OPTS['data_root'],
        split_name = 'train',
        max_items = GLOBAL_OPTS['train_dataset_size'],
        max_capt_len = GLOBAL_OPTS['max_capt_len'],
        verbose = GLOBAL_OPTS['verbose']
    )
    test_data = coco_data.COCODataSplit(
        GLOBAL_OPTS['coco_json'],
        GLOBAL_OPTS['data_root'],
        split_name = 'test',
        max_items = GLOBAL_OPTS['test_dataset_size'],
        max_capt_len = GLOBAL_OPTS['max_capt_len'],
        verbose = GLOBAL_OPTS['verbose']
    )
    val_data = coco_data.COCODataSplit(
        GLOBAL_OPTS['coco_json'],
        GLOBAL_OPTS['data_root'],
        split_name = 'val',
        max_items = GLOBAL_OPTS['val_dataset_size'],
        max_capt_len = GLOBAL_OPTS['max_capt_len'],
        verbose = GLOBAL_OPTS['verbose']
    )

    # Put aliases in dict so that we can iterate over splits
    split_objs = {
        'train': train_data,
        'test' : test_data,
        'val'  : val_data
    }
    split_names = {
        'train': GLOBAL_OPTS['train_dataset_fname'],
        'test' : GLOBAL_OPTS['test_dataset_fname'],
        'val'  : GLOBAL_OPTS['val_dataset_fname']
    }

    # For the caption/caplen files, we just clip out the path and extension of
    # the filename for each split and append _captions.json or _caplens.json

    for k, v in split_objs.items():
        print('Getting data for split <%s>' % str(k))
        v.create_split()

    wmap = word_map.WordMap()
    for k, v in split_objs.items():
        print('Updating word map with data from split <%s>' % str(k))
        split_captions = v.get_captions()
        for s in split_captions:
            wmap.update(s)

    wmap.generate()
    print('Generated word map with %d words' % len(wmap))
    if GLOBAL_OPTS['verbose']:
        print('Writing word map to file [%s]' % GLOBAL_OPTS['wordmap_fname'])
    wmap.save(GLOBAL_OPTS['wordmap_fname'])

    # Process the data in the splits
    for k, v in split_objs.items():
        print('Processing data for split <%s>' % str(k))
        coco_proc.process_coco_data_split(v, wmap.get_word_map(), split_names[k], split_name=str(k))
        if GLOBAL_OPTS['verbose']:
            print('Generated split <%s> containing %d items'  % (str(k), len(v)))


def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    # data arguments
    parser.add_argument('--data-root',
                        type=str,
                        default='/mnt/ml-data/datasets/COCO',
                        help='Path to data root'
                        )
    parser.add_argument('--coco-json',
                        type=str,
                        default='data/dataset_coco.json',     # karpathy splits
                        help='Path to json file to use for COCO test'
                        )
    # processing options
    parser.add_argument('--max-capt-len',
                        type=int,
                        default=48,
                        help='Maximum length of caption vector'
                        )
    # output file options
    parser.add_argument('--train-dataset-fname',
                        type=str,
                        default='hdf5/coco_train.h5',
                        help='Set name for train dataset file'
                        )
    parser.add_argument('--train-dataset-size',
                        type=int,
                        default=0,
                        help='Maximum size of train dataset. Value of 0 processes all data (default 0)'
                        )
    parser.add_argument('--test-dataset-fname',
                        type=str,
                        default='hdf5/coco_test.h5',
                        help='Set name for test dataset file'
                        )
    parser.add_argument('--wordmap-fname',
                        type=str,
                        default='hdf5/wordmap.json',
                        help='Filename to use for wordmap'
                        )
    parser.add_argument('--test-dataset-size',
                        type=int,
                        default=0,
                        help='Maximum size of test dataset. Value of 0 processes all data (default 0)'
                        )
    parser.add_argument('--val-dataset-fname',
                        type=str,
                        default='hdf5/coco_val.h5',
                        help='Set name for val dataset file'
                        )
    parser.add_argument('--val-dataset-size',
                        type=int,
                        default=0,
                        help='Maximum size of val dataset. Value of 0 processes all data (default 0)'
                        )
    parser.add_argument('--overwrite',
                        default=False,
                        action='store_true',
                        help='Overwrites files if they already exist'
                        )

    return parser


# Entry point
if __name__ == '__main__':
    parser = get_parser()
    opts   = vars(parser.parse_args())

    for k, v in opts.items():
        GLOBAL_OPTS[k] = v

    if GLOBAL_OPTS['verbose'] is True:
        print(' ---- GLOBAL OPTIONS [%s]---- ' % sys.argv[0])
        for k,v in GLOBAL_OPTS.items():
            print('\t [%s] : %s' % (str(k), str(v)))

    main()
