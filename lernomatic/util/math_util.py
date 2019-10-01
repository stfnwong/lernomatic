"""
MATH_UTIL
Various little math functions that come in handy

Stefan Wong 2019
"""

import numpy as np


def is_pow2_int(X:int) -> bool:
    return ((X & (X - 1)) == 0 and (X != 0))

def is_pow2_pos_int(X:int) -> bool:
    return (X > 0) and (X & (X - 1))


# Averaging
def moving_avg(X:np.ndarray, n:int=3) -> np.ndarray:
    r = np.cumsum(X, dtype=X.dtype)
    r[n:] = r[n:] - r[:-n]

    return r[n - 1:] / n

