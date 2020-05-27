"""
TEST UTILS

Stefan Wong 2020
"""

import torch

def get_device_id(device_id:int=0) -> int:
    if torch.cuda.is_available():
        print('Using devce cuda:%d' % device_id)
        return device_id
    return -1
