"""
COCO_DATASET
Dataset object for COCO data

Stefan Wong 2018
"""

import torch
from torch.utils.data import Dataset
import h5py

# debug
#from pudb import set_trace; set_trace()


class CaptionDataset(Dataset):
    def __init__(self, h5_file, **kwargs):
        self.h5_file   = h5_file
        self.transform = kwargs.pop('transform', None)
        self.verbose   = kwargs.pop('verbose', False)

        # load params from h5 file
        self.fp        = h5py.File(h5_file, 'r')
        self.imgs      = self.fp['images']
        self.captions  = self.fp['captions']
        self.caplens   = self.fp['caplens']
        self.cpi       = self.fp.attrs['capt_per_img']
        # display options
        self.disp_name = kwargs.pop('disp_name', None)

    def __repr__(self):
        return 'CaptionDataset'

    def __str__(self):
        s = []
        s.append('CaptionDataset\n')
        s.append('%d items, %d captions per image\n' % (len(self), self.cpi))

        return ''.join(s)

    def __del__(self):
        self.fp.close()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img = torch.FloatTensor(self.imgs[idx // self.cpi] / 255)
        if self.transform is not None:
            img = self.transform(img)
        # handle captions
        caption = torch.LongTensor(self.captions[idx])
        caplen  = torch.LongTensor(self.caplens[idx])

        if self.fp.attrs['split_name'] in ('train', 'TRAIN'):
            return (img, caption, caplen)
        else:
            # for validation or testing, also return all 'capt_per_img'
            # captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((idx // self.cpi) * self.cpi) : (((idx // self.cpi) * self.cpi) + self.cpi)]
            )

            return (img, caption, caplen, all_captions)
