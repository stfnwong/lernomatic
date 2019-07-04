"""
EX_CORNELL_DIALOG
Examples that use the Cornell Dialog Corpus

Stefan Wong 2019
"""


import argparse
import torch.nn as nn

from lernomatic.data.text import cornell_movie
from lernomatic.data.text import vocab
from lernomatic.data.text import qr_batch
from lernomatic.data.text import qr_dataset
# models and trainer
from lernomatic.models.text import seq2seq
from lernomatic.train.text import seq2seq_trainer


# debug
from pudb import set_trace; set_trace()

GLOBAL_OPTS = dict()


def main() ->None:
    corpus_lines_filename         = GLOBAL_OPTS['data_root'] + 'cornell_movie_dialogs_corpus/movie_lines.txt'
    corpus_conversations_filename = GLOBAL_OPTS['data_root'] + 'cornell_movie_dialogs_corpus/movie_conversations.txt'
    qr_pairs_csv_file             = 'data/cornell_corpus_out.csv'

    if GLOBAL_OPTS['vocab_infile'] is not None:
        mvocab.load(GLOBAL_OPTS['vocab_infile'])
    else:
        print('Preparing vocabulary...')
        mcorpus = cornell_movie.CornellMovieCorpus(
            corpus_lines_filename,
            corpus_conversations_filename,
            verbose=True
        )
        qr_pairs = mcorpus.extract_sent_pairs(max_length=GLOBAL_OPTS['max_qr_len'])
        mvocab = vocab.Vocabulary('Cornell Movie Vocabulary')
        for n, pair in enumerate(qr_pairs):
            print('Adding pair [%d/%d] to vocab' % (n+1, len(qr_pairs)), end='\r')
            mvocab.add_sentence(pair.query)
            mvocab.add_sentence(pair.response)
        print('\n Created new vocabulary of %d words' % len(mvocab))
        print('Pruning words that appear fewer than %d times' % GLOBAL_OPTS['min_word_freq'])
        mvocab.trim_freq(GLOBAL_OPTS['min_word_freq'])
        print('Created new vocabulary:')
        print(mvocab)

    ## FIXME: I think there is a problem here with the size of the vocab...
    #if len(mvocab) < 33021:
    #    print('Warning, vocab length (%d) < 33021' % len(mvocab))

    # Get some datasets (generate these with proc tool first)
    train_dataset = qr_dataset.QRDataset(
        GLOBAL_OPTS['train_dataset']
    )
    val_dataset = qr_dataset.QRDataset(
        GLOBAL_OPTS['val_dataset']
    )

    # get some models
    embedding_layer = nn.Embedding(len(mvocab), GLOBAL_OPTS['hidden_size'])

    encoder = seq2seq.EncoderRNN(
        GLOBAL_OPTS['hidden_size'],
        num_layers = GLOBAL_OPTS['enc_num_layers'],
        num_words  = len(mvocab),
        embedding = embedding_layer,
    )
    decoder = seq2seq.LuongAttenDecoderRNN(
        GLOBAL_OPTS['hidden_size'],
        len(mvocab),
        num_layers = GLOBAL_OPTS['dec_num_layers'],
        embedding = embedding_layer,
    )


    # get a trainer
    trainer = seq2seq_trainer.Seq2SeqTrainer(
        mvocab,
        encoder,
        decoder,
        # generic training options
        # TODO : add these to GLOBAL_OPTS
        batch_size = 16,
        num_epochs = 10,
        learning_rate = 4e-3,
        device_id = GLOBAL_OPTS['device_id'],

        # dataset options
        train_dataset = train_dataset,
        val_dataset = val_dataset
    )

    trainer.train()


    print('OK')


def get_parser():
    parser = argparse.ArgumentParser()
    # General opts
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        default=False,
                        help='Set verbose mode'
                        )
    # Device options
    parser.add_argument('--device-id',
                        type=int,
                        default=-1,
                        help='Set device id (-1 for CPU)'
                        )
    # vocab options
    parser.add_argument('--vocab-outfile',
                        type=str,
                        default=None,
                        help='If present, save the vocabulary to this file (default: None)'
                        )
    parser.add_argument('--vocab-infile',
                        type=str,
                        default=None,
                        help='If present, load the vocabulary from this file (default: None)'
                        )

    # model options
    parser.add_argument('--hidden-size',
                        type=int,
                        default=512,
                        help='Number of hidden layers in each model (default: 512)'
                        )
    parser.add_argument('--enc-num-layers',
                        type=int,
                        default=1,
                        help='Number of layers to use in encoder (default: 2)'
                        )
    parser.add_argument('--dec-num-layers',
                        type=int,
                        default=1,
                        help='Number of layers to use in decoder (default: 2)'
                        )
    # training options
    parser.add_argument('--batch-size',
                        type=int,
                        default=64,
                        help='Batch size to use for training (default: 64)'
                        )
    parser.add_argument('--num-epochs',
                        type=int,
                        default=32,
                        help='Number of epochs to train for (default: 32)'
                        )
    # TODO : learning rate, learning rate finder...

    # data options
    parser.add_argument('--data-root',
                        type=str,
                        default='/mnt/ml-data/datasets/',
                        help='Path to root of dataset'
                        )
    parser.add_argument('--min-word-freq',
                        type=int,
                        default=5,
                        help='Minimum number of times a word can occur before it is pruned from the vocabulary (default: 5)'
                        )
    parser.add_argument('--max-qr-len',
                        type=int,
                        default=20,
                        help='Maximum length in words that a query or response may be (default: 10)'
                        )
    # datasets
    parser.add_argument('--train-dataset',
                        type=str,
                        default='hdf5/cornell-movie-train.h5',
                        help='Path to training dataset (default: hdf5/cornell-movie-train.h5)'
                        )
    parser.add_argument('--val-dataset',
                        type=str,
                        default='hdf5/cornell-movie-val.h5',
                        help='Path to validation dataset (default: hdf5/cornell-movie-val.h5)'
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
            print('\t[%s] : %s' % (str(k), str(v)))

    main()
