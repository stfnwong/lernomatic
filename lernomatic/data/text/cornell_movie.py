"""
CORNELL_MOVIE
Preprocessing for the Cornell Movie Dialogs Corpus

Stefan Wong 2019
"""

import codecs
import json
from lernomatic.util import text_util
from lernomatic.data.text import qr_pair



class CornellMovieCorpus(object):
    """
    TODO : docstring
    """
    def __init__(self, movie_lines_file:str, movie_conv_file:str, **kwargs) -> None:
        self.lines:dict         = dict()
        self.conversations:list = list()

        # internal constants
        self.seperator = ' +++$+++ '
        # Field ids for the various fields in the corpus
        self.movie_lines_fields = ['lineID', 'characterID', 'movieID', 'character', 'text']
        self.movie_conversation_fields = ['character1ID', 'character2ID',  'movieID', 'utteranceIDs']

        # keyword args
        self.verbose : bool    = kwargs.pop('verbose', False)
        self.target_offset:int = kwargs.pop('target_offset', 1)
        self.delimiter:str     = kwargs.pop('delimiter', '\t')

        # un-escape the output delimiter
        self.delimiter = str(codecs.decode(self.delimiter, 'unicode_escape'))

        # load the files
        self._load_lines(movie_lines_file)
        self._load_conversations(movie_conv_file)

    def __repr__(self) -> str:
        return 'CornellMovieCorpus'

    def __str__(self) -> str:
        return 'CornellMovieCorpus [%d words, %d conversations]' % (len(self.lines), len(self.conversations))

    def _load_lines(self, filename:str, encoding:str='iso-8859-1') -> None:
        self.lines = dict()
        with open(filename, 'r', encoding=encoding) as fp:
            for line in fp:
                values = line.split(self.seperator)
                line_obj = {}
                for n, field in enumerate(self.movie_lines_fields):
                    line_obj[field] = values[n]
                self.lines[line_obj['lineID']] = line_obj

        if self.verbose:
            print('Read %d lines from file [%s]' % (len(self.lines), str(filename)))

    def _load_conversations(self, filename:str, encoding:str='iso-8859-1') -> None:
        self.conversations = list()
        with open(filename, 'r', encoding=encoding) as fp:
            for line in fp:
                values = line.split(self.seperator)
                conv_obj = {}
                for n, field in enumerate(self.movie_conversation_fields):
                    conv_obj[field] = values[n]

                # convert string to list
                line_ids = eval(conv_obj['utteranceIDs'])
                # re-assemble lines
                conv_obj['lines'] = []
                for line_id in line_ids:
                    conv_obj['lines'].append(self.lines[line_id])
                self.conversations.append(conv_obj)

        if self.verbose:
            print('Read %d conversations from file [%s]' % (len(self.conversations), str(filename)))

    def get_num_conversations(self) -> int:
        return len(self.conversations)

    def get_num_lines(self) -> int:
        return len(self.lines)

    def get_param_dict(self) -> dict:
        return {
            'lines' : self.lines,
            'conversations' : self.conversations
        }

    def set_param_dict(self, param:dict) -> None:
        self.lines = param['lines']
        self.conversations = param['conversations']

    def save(self, filename:str) -> None:
        params = self.get_param_dict()
        json.dumps(filename)

    def load(self, filename:str) -> None:
        params = json.loads(filename)
        self.set_param_dict(params)

    def extract_sent_pairs(self, max_length:int=0) -> list:
        qr_pairs = []
        for n, conv in enumerate(self.conversations):
            if self.verbose:
                print('Extracting Q/R pair from conversation [%d/%d]' %\
                      (n+1, len(self.conversations)), end='\r'
                )

            # since the last line has no answer we ignore it
            for i in range(len(conv['lines']) - 1):
                input_line  = conv['lines'][i]['text'].strip()
                target_line = conv['lines'][i+self.target_offset]['text'].strip()

                # filter out samples where the list is empty
                if input_line and target_line:
                    pair = qr_pair.QRPair(
                        query=text_util.normalize_string(input_line),
                        response=text_util.normalize_string(target_line)
                    )
                    # filter if required
                    if max_length > 0:
                        if (pair.query_len() >= max_length) or (pair.response_len() >= max_length):
                            continue
                    qr_pairs.append(pair)

        if self.verbose:
            print('\n  done. Extracted  %d sentence pairs total' % len(qr_pairs))

        return qr_pairs
