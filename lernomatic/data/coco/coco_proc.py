"""
PROC
COCO dataset processing tools

Stefan Wong 2019
"""

import h5py
import random
import numpy as np
from tqdm import tqdm
import cv2

# debug
#from pudb import set_trace; set_trace()


def process_coco_data_split(split_data, word_map, fname, **kwargs):
    """
    PROCESS_COCO_DATA_SPLIT
    Process the data in the COCOSplitData split into an HDF5 file
    """

    # deal with keyword args
    seed       = kwargs.pop('seed', None)
    pixel_max  = kwargs.pop('pixel_max', 255)
    split_name = kwargs.pop('split_name', None)

    # Process data and store in hdf5 file
    with h5py.File(fname, 'w') as fp:
        # Save split attributes
        fp.attrs['capt_per_img'] = split_data.capt_per_img
        fp.attrs['max_capt_len'] = split_data.max_capt_len
        if split_name is not None:
            fp.attrs['split_name'] = split_name

        images = fp.create_dataset(
            'images',
            (len(split_data), 3, split_data.img_dim, split_data.img_dim),
            dtype='uint8')
        caption_data = fp.create_dataset(
            'captions',
            (len(split_data), split_data.max_capt_len),
            dtype='int32'
        )
        caplens = fp.create_dataset(
            'caplens',
            (len(split_data), 1),
            dtype='int32',
        )
        if seed is not None:
            np.random.seed(seed)

        # Process all captions
        for n, (impath, imcap) in enumerate(tqdm(split_data, unit='samples', total=len(split_data))):
            if len(imcap) < split_data.capt_per_img:
                captions = imcap + [np.random.choice(imcap) for _ in range(split_data.capt_per_img - len(imcap))]
            else:
                captions = random.sample(imcap, k=split_data.capt_per_img)
            if len(captions) != split_data.capt_per_img:
                raise ValueError('len(captions) (%d) != capt_per_img (%d)' %\
                                    (len(captions), split_data.capt_per_img))
            # read images
            img = cv2.imread(impath)
            if len(img.shape) == 2:     # grayscale
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)
            #img = imresize(img, (256, 256))
            img = cv2.resize(img, (split_data.img_dim, split_data.img_dim), interpolation=cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1)
            assert img.shape == (3, split_data.img_dim, split_data.img_dim)
            assert np.max(img) <= pixel_max

            # save image to HDF5
            images[n] = img
            enc_captions = []
            enc_caplens = []
            for j, c in enumerate(captions):
                # encode captions
                enc_c = [word_map['<start>']] +\
                    [word_map.get(word, word_map['<unk>']) for word in c] +\
                    [word_map['<end>']]
                if len(enc_c) > split_data.max_capt_len:
                    enc_c = enc_c[0 : split_data.max_capt_len]
                    enc_c[-1] = word_map['<end>']
                elif len(enc_c) < split_data.max_capt_len:
                    # pad out the rest of the caption
                    enc_c += [word_map['<pad>']] * (split_data.max_capt_len - len(c) - 2)
                # find caption lengths
                c_len = len(c) + 2
                enc_captions.append(enc_c)
                enc_caplens.append(c_len)

            #caption_data[n][:] = enc_c[:]
            caption_data[n] = enc_c
            caplens[n]  = c_len


#TODO : a version that stores captions in JSON format.
#
