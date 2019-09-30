"""
BUILD_CUDA
Build CUDA components

"""

import os
import torch
from torch.utils.ffi import create_extension

# TODO : software versions later...
#sources = ['cuda/stn/stn.c']
#headers = ['cuda/stn/stn.h']
#defines = []

if torch.cuda.is_available():
    sources = ['cuda/stn/stn_cuda.c']
    headers = ['cuda/stn/stn_cuda.h']
    defines = [('WITH_CUDA', None)]


this_file = os.path.dirname(os.path.realpath(__file__))

extra_objects = ['cuda/stn/stn_cuda.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

# Create the foreign function interface
extension_path = 'lernomatic.ext.stn'
ffi = create_extension(
    extension_path,
    headers = headers,
    sources = sources,
    define_macros = defines,
    relative_to = __file__,     # TODO : this might not be right
    with_cuda = True,
    extra_objects = extra_objects
)

if __name__ == '__main__':
    ffi.build()
