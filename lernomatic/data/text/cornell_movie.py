"""
CORNELL_MOVIE
Preprocessing for the Cornell Movie Dialogs Corpus

"""

import json


class QRPair(object):
    def __init__(self, query:str='', resp:str='') -> None:
        """
        Query/Reponse pair
        """
        self.query   = query
        self.reponse = resp

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




class CornellMovieCorpus(object):
    def __init__(self, **kwargs) -> None:
        self.lines:dict         = dict()
        self.conversations:list = list()      # TODO : required?

        # keyword args
        self.verbose : bool    = kwargs.pop('verbose', False)
        self.target_offset:int = kwargs.pop('target_offset', 1)

        # internal constants
        self.sep_string = ' +++$+++ '
        # Field ids for the various fields in the corpus
        self.movie_lines_fields = ['lineID', 'characterID', 'movieID', 'character', 'text']
        self.movie_conversation_fields = ['character1ID', 'character2ID',  'movieID', 'utteranceIDs']

    def __repr__(self) -> str:
        return 'CornellMovieCorpus'

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

    def load_lines(self, filename:str, encoding:str='iso-8859-1') -> None:
        self.lines = dict()
        with open(filename, 'r', encoding=encoding) as fp:
            for line in fp:
                values = line.split(self.sep_string)
                line_obj = {}
                for n, field in enumerate(self.movie_lines_fields):
                    line_obj[field] = values[n]
                self.lines[line_obj['lineID']] = line_obj

    def load_conversations(self, filename:str, encoding:str='iso-8859-1') -> None:
        if len(self.lines) == 0:
            if self.verbose:
                print('No lines in object, call load_lines() first')
            return

        self.conversations = list()
        with open(filename, 'r', encoding=encoding) as fp:
            for line in fp:
                values = line.split(self.sep_string)
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

    def extract_sent_pairs(self) -> None:
        if len(self.conversations) == 0:
            if self.verbose:
                print('No conversations in object, call load_conversations() first')
            return

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
