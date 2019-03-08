"""
VIS_DATA

"""

#def get_img_figure(n_subplots=1):

def plot_img(ax, img, **kwargs):
    title = kwargs.pop('title', 'Image')

    ax.imshow(img)
    ax.set_title(title)
