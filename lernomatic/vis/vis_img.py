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
    for px in range(1, num_x+1):
        for py in range(1, num_y+1):
            sub_ax = fig.add_subplot(num_subplots, px, py)
            ax.append(sub_ax)

    return (fig, ax)
