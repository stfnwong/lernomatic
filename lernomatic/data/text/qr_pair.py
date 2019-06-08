"""
QR_PAIR
Query/response pair

Stefan Wong 2019
"""

import csv
from lernomatic.util import text_util


class QRPair(object):
    def __init__(self, query:str='', response:str='') -> None:
        """
        Query/Reponse pair
        """
        self.query     = query
        self.response  = response
        # can adjust seperator for odd sentences
        self.seperator = ' '

    def __repr__(self) -> str:
        return 'QRPair'

    def __str__(self) -> str:
        s = []
        s.append('QRPair\n')
        s.append('\tQ: [%s]\n' % str(self.query))
        s.append('\tR: [%s]\n' % str(self.response))
        return ''.join(s)

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

    def to_list(self) -> list:
        return [self.query, self.response]

    def to_tuple(self) -> tuple:
        return (self.query, self.response)

    def query_to_list(self) -> list:
        return self.query.split(self.seperator)

    def response_to_list(self) -> list:
        return self.response.split(self.seperator)


def qr_pair_proc_from_csv(filename:str,
                          encoding:str='utf-8',
                          delimiter:str='\t',
                          max_length:int=0,
                          verbose:bool=False) -> list:
    num_filtered = 0
    with open(filename, 'r', encoding=encoding) as fp:
        lines = fp.read().strip().split('\n')

    qr_pairs = list()
    for n, line in enumerate(lines):
        if verbose:
            print('Processing line [%d/%d] from file [%s]' % (n+1, len(lines), str(filename)), end='\r')
        split_line = [text_util.normalize_string(s) for s in line.split(delimiter)]
        pair = QRPair(
            query=split_line[0],
            response=split_line[1]
        )
        # filter out pairs that are too long
        if max_length > 0:
            if (pair.query_len() >= max_length) or (pair.response_len() >= max_length):
                num_filtered += 1
                continue

        qr_pairs.append(pair)

    if verbose:
        print('\n Processed %d Query/Response pairs' % len(qr_pairs))
        print('Filtered %d pairs for being longer than %d words' % (num_filtered, max_length))

    return qr_pairs


def qr_pairs_to_csv(filename:str,
                    qr_pairs:list,
                    encoding:str='utf-8',
                    delimiter:str='\t',
                    verbose:bool = False) -> None:
    """
    qr_pairs_to_csv()
    Write a list of QRPairs to a *.csv file
    """
    with open(filename, 'w', encoding=encoding) as fp:
        writer = csv.writer(fp, delimiter=delimiter, lineterminator='\n')
        for n, pair in enumerate(qr_pairs):
            if verbose:
                print('Writing pair [%d/%d] to file [%s]' % \
                        (n+1, len(qr_pairs), str(filename)), end='\r'
                )
            writer.writerow(pair.to_tuple())

        if verbose:
            print('\n done. Wrote %d pairs to disk' % len(qr_pairs))
