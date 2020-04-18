"""
TEST UTILS

Stefan Wong 2020
"""

import torch

def get_device_id() -> int:
    if torch.cuda.is_available():
        print('Using cuda:0')
        return 0
    return -1
