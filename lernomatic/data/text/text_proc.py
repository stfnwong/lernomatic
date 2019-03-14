"""
TEXT_PROC
Process a text dataset

Stefan Wong 2019
"""

import h5py
from tqdm import tqdm
from lernomatic.data.text import corpus
from lernomatic.data.text import word_map

# debug
from pudb import set_trace; set_trace()


class TextHDF5Proc(object):
    """
    TextHDF5Proc
    Process text into an HDF5 dataset

    Args:
        eos_chars:
            List of characters in source text that should be taken
            to indicate end-of-sentences. These characters will be
            replaced with the <EOS> token.

    """
    def __init__(self, **kwargs) -> None:
        self.verbose   : bool = kwargs.pop('verbose', False)
        self.eos_chars : list = kwargs.pop('eos_chars', ['.', ';'])
        self.eos_token : str  = kwargs.pop('eos_token', '<end>')
        self.data_key  : str  = kwargs.pop('data_key', 'text')

    def __repr__(self) -> str:
        return 'TextHDF5Proc'

    def proc(self, in_filename : str, out_filename : str) -> None:
        raise NotImplemented('This method should be implemented in subclass')


class TextWordLevelProc(TextHDF5Proc):
    def __init__(self, wmap : word_map.WordMap, **kwargs) -> None:
        self.wmap    = wmap
        self.len_key = kwargs.pop('len_key', 'lengths')
        super(TextWordLevelProc, self).__init__(**kwargs)

    def __repr__(self):
        return 'TextWordLevelProc'

    def proc(self, in_filename : str, out_filename : str) -> None:
        """
        proc()

        Args:
            in_filename: Text file to read from
            out_filename : HDF5 file to write to
        """
        # Find number of words in file
        if self.verbose:
            print('Counting words in file [%s]' % str(in_filename))
        with open(in_filename, 'r') as text_fp:
            max_word_len = 0
            num_words = 0
            num_lines = 0
            for line in tqdm(text_fp, unit='lines'):
                num_lines += 1
                words = line.split()
                num_words += len(words)
                if len(words) > 0 and (len(max(words)) > max_word_len):
                    max_word_len = len(max(words))

        with h5py.File(out_filename, 'w') as fp:
            #dt = h5py.special_dtype(vlen=str)
            word_dataset = fp.create_dataset(self.data_key, (num_words, max_word_len), dtype=int)
            word_dataset.attrs['max_word_len'] = max_word_len
            #wlen_dataset = fp.create_dataset(self.len_key, (num_words, 1), dtype=int)

            if self.verbose:
                print('Procesing %d lines in file [%s]' % (num_lines, str(in_filename)))
            # iterate over the file and place characters into dataset
            word_idx = 0
            with open(in_filename, 'r') as text_fp:
                for line in tqdm(text_fp, unit='lines', total=num_lines):
                    words = line.split()
                    for w in words:
                        word_dataset[word_idx] = self.wmap.lookup_word(w)
                        #wlen_dataset[word_idx] = len(w)
                        word_idx += 1



class TextCharLevelProc(TextHDF5Proc):
    def __init__(self, **kwargs) -> None:
        super(TextCharLevelProc, self).__init__(**kwargs)

    def __repr__(self):
        return 'TextCharLevelProc'

    def proc(self, in_filename : str, out_filename : str) -> None:
        """
        proc()

        Args:
            in_filename: Text file to read from
            out_filename : HDF5 file to write to
        """

        # Find number of characters in file
        if self.verbose:
            print('Counting characters in file [%s]' % str(in_filename))
        with open(in_filename, 'r') as text_fp:
            num_chars = 0
            num_lines = 0
            for line in tqdm(text_fp, unit='lines'):
                num_lines += 1
                words = line.split()
                for w in words:
                    for c in w:
                        num_chars += 1

        with h5py.File(out_filename, 'w') as fp:
            dt = h5py.special_dtype(vlen=str)
            text_dataset = fp.create_dataset(self.data_key, (num_chars, 8), dtype=dt)

            if self.verbose:
                print('Procesing %d lines in file [%s]' % (num_lines, str(in_filename)))
            # iterate over the file and place characters into dataset
            char_idx = 0
            with open(in_filename, 'r') as text_fp:
                for line in tqdm(text_fp, unit='lines', total=num_lines):
                    words = line.split()
                    for w in words:
                        for c in w:
                            text_dataset[char_idx] = c
                            char_idx += 1
