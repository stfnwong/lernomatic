"""
FILE_UTIL
File-handling utils

Stefan Wong 2019
"""

import os

def get_file_paths(path_root:str, verbose:bool=False) -> list:
    out_paths = []

    for root, dirs, fnames in os.walk(path_root):
        fnames = sorted(fnames)

        for f in fnames:
            inp_path = os.path.abspath(root)
            file_path = os.path.join(inp_path, fname)

            if fname.endswith('.png') or fname.endswith('.jpg'):
                out_paths.append(file_path)

        break       # don't go into subfolders

    return out_paths
