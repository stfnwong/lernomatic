"""
LERNOMATIC SETUP
Setup files for Lernomatic
"""

import re
from setuptools import setup, find_packages

def cleanup(x):
    x = x.strip()               # whitespace
    x = re.sub(r' *#.*', '', x) # comments

    return x

def to_list(buf):
    return list(filter(None, map(cleanup, buf.splitlines())))

install_requirements = to_list("""
    tqdm
    torch
    torchvision
    matplotlib
    h5py
    pillow
    numpy >=1.12
""")

setup(
    name = "lernomatic",
    version = "0.333333",

    install_requires = install_requirements,
    python_requires = '>=3.4',

    test_suite = 'test',

    description = 'Lernomatic',
    author = 'Stefan Wong',
    author_email = 'stfnwong@gmail.com',

    zip_safe = False,
)
