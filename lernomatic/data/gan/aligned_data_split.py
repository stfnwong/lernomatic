"""
ALIGNED_DATA_SPLIT

Stefan Wong 2019
"""

import json
from lernomatic.data import data_split


class AlignedDataSplit(object):
    def __init__(self, split_name:str='train') -> None:
        self.split_name = split_name
        self.a_data_paths:list = []
        self.b_data_paths:list = []
        self.a_ids:list        = None
        self.b_ids:list        = None
        self.has_ids:bool      = False
        self.idx:int           = 0

    def __repr__(self) -> str:
        return 'AlignedDataset <%s> (%d items)' % (self.split_name, len(self))

    def __len__(self) -> int:
        return len(self.a_data_paths)

    def __iter__(self) -> 'AlignedDataSplit':
        self.idx = 0
        return self

    def __next__(self) -> tuple:
        if self.idx >= len(self):
            raise StopIteration

        a_path = self.a_data_paths[self.idx]
        b_path = self.b_data_paths[self.idx]

        if self.has_ids:
            a_id = self.a_ids[self.idx]
            b_id = self.b_ids[self.idx]
        else:
            a_id = None
            b_id = None
        self.idx += 1

        return (a_path, a_id, b_path, b_id)

    def __getstate__(self) -> dict:
        return {
            'split_name'   : self.split_name,
            'a_data_paths' : self.a_data_paths,
            'b_data_paths' : self.b_data_paths,
            'a_ids'        : self.a_ids,
            'b_ids'        : self.b_ids,
            'has_ids'      : self.has_ids,
        }

    def __setstate__(self, state:dict) -> None:
        self.split_name   = state['split_name']
        self.a_data_paths = state['a_data_paths']
        self.b_data_paths = state['b_data_paths']
        self.a_ids        = state['a_ids']
        self.b_ids        = state['b_ids']
        self.has_ids      = state['has_ids']

    def add_paths(self, a_path:str, b_path:str) -> None:
        self.a_data_paths.append(a_path)
        self.b_data_paths.append(b_path)

    def add_ids(self, a_id:int, b_id:int) -> None:
        self.has_ids = True
        self.a_ids.append(a_id)
        self.b_ids.append(b_id)

    def add_paths_ids(self, a_path:str, a_id:int, b_path:str, b_id:int) -> None:
        self.add_paths(a_path, b_path)
        self.add_ids(a_id, b_id)

    def save(self, fname:str) -> None:
        param = self.__getstate__()
        with open(fname, 'w') as fp:
            json.dump(param, fp)

    def load(self, fname:str) -> None:
        with open(fname, 'r') as fp:
            param = json.load(fp)
        self.__setstate__(param)



# ======== Datasplitter ======== #
class AlignedDatasetSplitter(data_split.DataSplitter):
    def __init__(self, data_root:str, **kwargs) -> None:
        self.data_root:str = data_root
        self.a_train_path:str = kwargs.pop('a_train_path', 'trainA')
        self.a_test_path:str  = kwargs.pop('a_test_path', 'testA')
        self.b_train_path:str = kwargs.pop('b_train_path', 'trainB')
        self.b_test_path:str  = kwargs.pop('b_test_path', 'testB')

        super(AlignedDataSplitter, self).__init__(**kwargs)

    def __repr__(self) -> str:
        return 'AlignedDataSplitter'

    def gen_splits(self) -> list:
        pass
