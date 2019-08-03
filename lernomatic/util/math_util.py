"""
MATH_UTIL
Various little math functions that come in handy

Stefan Wong 2019
"""

def is_pow2_int(X:int) -> bool:
    return ((X & (X - 1)) == 0 and (X != 0))

def is_pow2_pos_int(X:int) -> bool:
    return (X > 0) and (X & (X - 1))
