"""
QR_DATASET
Query/Reponse dataset wrapper

Stefan Wong 2019
"""

import h5py
from torch.utils.data import Dataset
from lernomatic.data.text import qr_batch
from lernomatic.data.text import vocab


class QRDataset(Dataset):
    def __init__(self, filename:str, **kwargs) -> None:
        self.fp = h5py.File(filename, 'r')
        # set dataset names
        self.query_name:str           = kwargs.pop('query_name', 'query')
        self.response_name:str        = kwargs.pop('response_name', 'reponse')
        self.query_length_name:str    = kwargs.pop('query_length_name', 'qlength')
        self.response_length_name:str = kwargs.pop('response_length_name', 'rlength')

    def __repr__(self) -> str:
        return 'QRDataset'

    def __del__(self) -> None:
        self.fp.close()

    def __len__(self) -> int:
        return len(self.fp[self.feature_name])

    def __getitem__(self, idx) -> tuple:
        if idx > len(self):
            raise IndexError('idx %d out of range (%d)' % (idx, len(self)))

        query           = torch.LongTensor(self.fp[self.query_name][idx][:])
        response        = torch.LongTensor(self.fp[self.response_name][idx][:])
        query_length    = torch.LongTensor(self.fp[self.query_length_name][idx][:])
        response_length = torch.LongTensor(self.fp[self.reposnse_length_name][idx][:])

        return (query, query_length, response, response_length)
        #return feature, label
