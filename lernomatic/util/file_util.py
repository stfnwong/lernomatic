"""
FILE_UTIL
File-handling utils

Stefan Wong 2019
"""

import os


def get_file_paths(path_root:str, **kwargs) -> list:
    skip_subdir:bool = kwargs.pop('skip_subdir', False)
    verbose:bool     = kwargs.pop('verbose', False)
    extensions:list  = kwargs.pop('extensions', ['jpg', 'png'])
    full_path:bool   = kwargs.pop('full_path', True)

    out_paths = []
    for dirname, subdir, filelist in os.walk(path_root):
        if len(filelist) > 0:
            if verbose:
                print('Found %d files in path [%s]' % (len(filelist), str(dirname)))

            for fname in filelist:
                for ext in extensions:
                    if ext in fname:
                        if full_path:
                            out_paths.append(str(dirname) + '/' + str(fname))
                        else:
                            out_paths.append(fname)
                        break

        # don't go into subfolders
        if skip_subdir:
            break

    return out_paths


def check_valid_paths(path_list:list) -> tuple:
    valid_paths = list()
    invalid_paths = list()

    for path in path_list:
        if os.path.isfile(path):
            valid_paths.append(path)
        else:
            invalid_paths.append(path)

    return (valid_paths, invalid_paths)

