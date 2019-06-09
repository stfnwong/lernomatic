""""
QR_PROC
Processor for Query/Response data

Stefan Wong 2019
"""

from tqdm import tqdm
import h5py
import numpy as np
from lernomatic.data.text import qr_split
from lernomatic.data.text import vocab


class QRDataProc(object):
    """
    QRDataProc

    Arguments:

        verbose: (bool)
            Sets verbose mode. (default: False)

        vec_len: (int)
            Sets the length that text vectors should be when they are written
            to file. Vectors will be either truncated to fit, or padded with
            the pad token in the supplied vocabulary. (default: 20)

        query_dataset_name: (str)
            Name to use for the query dataset (default: query)

        response_dataset_name: (str)
            Name to use for the response dataset (default: response)

        q_length_dataset_name: (str)
            Name to use for dataset containing true query lengths (default: qlength)

        r_length_dataset_name: (str)
            Name to use for dataset containing true response lengths (default: rlength)

    """
    def __init__(self, **kwargs) -> None:
        self.verbose:bool              = kwargs.pop('verbose', False)
        self.vec_len:int               = kwargs.pop('vec_len', 20)
        self.delimiter:str             = kwargs.pop('delimiter', ' ')
        self.query_dataset_name:str    = kwargs.pop('query_dataset_name', 'query')
        self.response_dataset_name:str = kwargs.pop('response_dataset_name', 'response')
        self.q_length_dataset_name:str = kwargs.pop('q_length_dataset_name', 'qlength')
        self.r_length_dataset_name:str = kwargs.pop('r_length_dataset_name', 'rlength')

    def __repr__(self) -> str:
        return 'QRDataProc'

    def proc(self,
             voc:vocab.Vocabulary,
             data_split:qr_split.QRDataSplit,
             filename:str) -> None:
        with h5py.File(filename, 'w') as fp:
            # text data
            queries = fp.create_dataset(
                self.query_dataset_name,
                (len(data_split), self.vec_len),
                dtype=np.int32
            )
            responses = fp.create_dataset(
                self.response_dataset_name,
                (len(data_split), self.vec_len),
                dtype=np.int32
            )

            # True lengths of each vector
            r_lengths = fp.create_dataset(
                self.r_length_dataset_name,
                (len(data_split),1),
                dtype=np.int32
            )
            q_lengths = fp.create_dataset(
                self.q_length_dataset_name,
                (len(data_split),1),
                dtype=np.int32
            )
            queries.attrs['num_words'] = len(voc)
            queries.attrs['vec_len'] = self.vec_len

            # TODO : regarding the use of <sos>, in some examples we just set
            # the initial hidden state of the decoder to be all <sos>, in which
            # case I don't think we need to insert that token  during
            # pre-processing (ie: here).
            for elem_idx, pair in enumerate(tqdm(data_split, unit='Q/R pairs')):
                q_lengths[elem_idx] = pair.query_len()
                r_lengths[elem_idx] = pair.response_len()

                # convery query
                enc_query = [voc.lookup_word(w) for w in pair.query_to_list()] +\
                            [voc.get_eos()]
                if len(enc_query) < self.vec_len:
                    enc_query.extend([voc.get_pad()] * (self.vec_len-len(enc_query)))
                elif len(enc_query) > self.vec_len:
                    q_lengths[elem_idx] = vec_len
                    enc_query = enc_query[0 : self.vec_len-1]
                    enc_query.extend([voc.get_eos()])
                queries[elem_idx] = enc_query

                # convert response
                enc_response = [voc.lookup_word(w) for w in pair.response_to_list()] +\
                            [voc.get_eos()]
                if len(enc_response) < self.vec_len:
                    enc_response.extend([voc.get_pad()] * (self.vec_len-len(enc_response)))
                elif len(enc_response) > self.vec_len:
                    r_lengths[elem_idx] = vec_len
                    enc_response = enc_response[0 : self.vec_len-1]
                    enc_response.extend([voc.get_eos()])
                responses[elem_idx] = enc_response
