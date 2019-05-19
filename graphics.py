import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import numpy as np
import math
from code import *

#graphics
def plot_images( ims, cmap=cm.Greys, max_rows_cols=10, figsize=(12, 12) ):
    n        = max(2,min(max_rows_cols, int(math.sqrt(len(ims))) ))
    fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=figsize)
    n_figs   = min(len(ims),n**2)
    for i in range(n_figs):
        ax = axs[i // n, i % n]
        ax.imshow( ims[i], cmap=cm.Greys)
        ax.axis('off')
    plt.tight_layout()


def draw_images_from_dataset(dataset, size:int, line_width:int, linetype:int, nb=1000, nrows=8, compressed_drawings=True, figsize=(12, 12)):
    print("reading and converting compressed drawings to images")
    compressed = [dataset.x[i] for i in np.random.randint(0,len(dataset.x),nb) ]
    ims = drawings2images(compressed, size, line_width, linetype, compressed_drawings) 
    print(f"plot images : {len(ims)}")
    plot_images( ims, cm.Greys, max_rows_cols=nrows, figsize=figsize )
