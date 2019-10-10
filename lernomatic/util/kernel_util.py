"""
KERNEL UTIL
Utils for Kernel Density Estimation

Stefan Wong 2019
"""

import numpy as np


# 'Silverman' bandwidth
def silverman_bw(data:np.ndarray) -> float:
    return np.std(data) * np.power((4 / (3 * np.prod(data.shape))), 1/5)


# Normal density function
def normal_phi(X:np.ndarray) -> np.ndarray:
    return np.square(np.exp(-0.5 * X) / np.sqrt(2 * np.pi))


# h is bandwidth parameter
def kernel_elem(X:float, data:float, h:float) -> np.ndarray:
    return np.mean(normal_phi((X - data) / h) / h)


def kernel(domain:np.ndarray, data:np.ndarray, h:float) -> np.ndarray:
    return np.fromiter((kernel_elem(domain, xi, h) for xi in data), data.dtype)


def kde(data:np.ndarray) -> np.ndarray:
    # silverman bandwidth
    # NOTE: I think I actually want a much larger bandwidth here
    h = silverman_bw(data)

    # evaluate the kernel function over this range
    domain = np.linspace(0, len(data), len(data))

    return kernel(domain, data, h)
