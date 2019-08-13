"""
VIS_IMG
Helper functions for visualizing images

Stefan Wong 2019
"""

import numpy as np
import matplotlib.pyplot as plt



def get_grid_subplots(num_x:int, num_y:int=0) -> tuple:
    if num_y == 0:
        num_y = num_x

    fig = plt.figure()
    ax = []

    num_subplots = num_x * num_y
    for p in range(1, num_subplots+1):
        sub_ax = fig.add_subplot(num_x, num_y, p)
        ax.append(sub_ax)

    return (fig, ax)
