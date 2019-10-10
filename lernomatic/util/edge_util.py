"""
EDGE_UTIL
Simple edge detection stuff

Stefan Wong 2019
"""


import numpy as np


# NOTE: these are all 1D, can generalize to higher dims later when needed

def laplace(X:np.ndarray) -> np.ndarray:
    k = np.array([-1, 4, -1])
    Xs = np.convolve(np.asarray(X), k)

    return Xs

def sobel(X:np.ndarray, offset:float=0.0) -> np.ndarray:
    k = np.array([-1, 0, 1])
    Xs = np.convolve(np.asarray(X + offset), k)

    return Xs

