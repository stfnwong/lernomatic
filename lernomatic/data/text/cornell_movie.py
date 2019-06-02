"""
CORNELL_MOVIE
Preprocessing for the Cornell Movie Dialogs Corpus

"""

import codecs
import csv
import json
import re
import unicodedata


def unicode_to_ascii(s:str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalize_string(s:str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class QRPair(object):
    def __init__(self, query:str='', response:str='') -> None:
        """
        Query/Reponse pair
        """
        self.query     = query
        self.response  = response
        self.seperator = ' '   # on the off chance that the sentence has some unusual seperator

    def __repr__(self) -> str:
        return 'QRPair'

    def __str__(self) -> str:
        return 'QRPair [%s : %s]' % (str(self.query), str(self.response))

    def __eq__(self, other:'QRPair') -> bool:
        if self.query != other.query:
            return False
        if self.response != other.response:
            return False
        return True

    def query_len(self) -> int:
        return len(self.query.split(self.seperator))

    def response_len(self) -> int:
        return len(self.response.split(self.seperator))


def qr_pair_proc(filename:str, **kwargs) -> list:
    verbose:bool  = kwargs.pop('verbose', False)
    encoding:str  = kwargs.pop('encoding', 'utf-8')
    seperator:str = kwargs.pop('seperator', '\t')
    max_length:int = kwargs.pop('max_length', 10)

    num_filtered = 0

    #lines = []
    with open(filename, 'r', encoding=encoding) as fp:
        lines = fp.read().strip().split('\n')

    qr_pairs = list()
    for n, line in enumerate(lines):
        if verbose:
            print('Processing line [%d/%d]' % (n+1, len(lines)), end='\r')
        split_line = [normalize_string(s) for s in line.split(seperator)]
        pair = QRPair(
            query=split_line[0],
            response=split_line[1]
        )
        # filter out pairs that are too long
        if (pair.query_len() > max_length) or (pair.response_len() > max_length):
            num_filtered += 1
        else:
            qr_pairs.append(pair)

    if verbose:
        print('\n Processed %d Query/Response pairs' % len(qr_pairs))
        print('Filtered %d pairs for being longer than %d words' % (num_filtered, max_length))

    return qr_pairs


class CornellMovieCorpus(object):
    def __init__(self, movie_lines_file:str, movie_conv_file:str, **kwargs) -> None:
        self.lines:dict         = dict()
        self.conversations:list = list()
        self.qa_pairs:list      = list()    # question/answer pairs

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

    def _load_lines(self, filename:str, encoding:str='iso-8859-1') -> None:
        self.lines = dict()
        with open(filename, 'r', encoding=encoding) as fp:
            for line in fp:
                values = line.split(self.seperator)
                line_obj = {}
                for n, field in enumerate(self.movie_lines_fields):
                    line_obj[field] = values[n]
                self.lines[line_obj['lineID']] = line_obj

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

    def extract_sent_pairs(self) -> None:

        self.qa_pairs = []
        for n, conv in enumerate(self.conversations):

            if self.verbose:
                print('Extracting sentence pair from conversation [%d/%d]' %\
                      (n+1, len(self.conversations)), end='\r'
                )

            # since the last line has no answer we ignore it
            for i in range(len(conv['lines']) - 1):
                input_line  = conv['lines'][i]['text'].strip()
                target_line = conv['lines'][i+self.target_offset]['text'].strip()

                # filter out samples where the list is empty
                if input_line and target_line:
                    self.qa_pairs.append([input_line, target_line])

        if self.verbose:
            print('\n  done. Extracted  %d sentence pairs total' % len(self.qa_pairs))

    def write_csv(self, filename:str, encoding:str='utf-8') -> None:
        if len(self.qa_pairs) == 0:
            if len(self.conversations) != 0:
                self.extract_sent_pairs()
            else:
                if self.verbose:
                    print('No qa_pairs in object, run extract_sent_pairs() first')
                return

        with open(filename, 'w', encoding=encoding) as fp:
            writer = csv.writer(fp, delimiter=self.delimiter, lineterminator='\n')
            for n, pair in enumerate(self.qa_pairs):
                if self.verbose:
                    print('Writing pair [%d/%d] to file [%s]' % \
                          (n+1, len(self.qa_pairs), str(filename)), end='\r'
                    )
                writer.writerow(pair)

            if self.verbose:
                print('\n done. Wrote %d pairs to disk' % len(self.qa_pairs))
