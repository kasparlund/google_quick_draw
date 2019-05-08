import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt 
import numpy as np
import math

#graphics
def plot_images( ims, cmap=cm.Greys, max_rows_cols=10 ):
    n        = max(2,min(max_rows_cols, int(math.sqrt(len(ims))) ))
    fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
    n_figs   = min(len(ims),n**2)
    for i in range(n_figs):
        ax = axs[i // n, i % n]
        ax.imshow( ims[i], cmap=cm.Greys)
        ax.axis('off')
    plt.tight_layout()