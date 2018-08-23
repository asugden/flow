import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os

# This should really be removed
import pool

from .misc import colors
from . import outfns


def open():
    """
    Set the size of the output image. Width can be full, half, or an
    integer. Height can be full or an integer.
    """

    style()

    # Set the width of the figure to be half-width if the half-width flag is true
    # Initialize the figure and axis
    fig = plt.figure(figsize=(14.35, 6))
    ax = plt.subplot(111)

    return fig, ax

def style(sz=20):
    """
    Set the style of a graph appropriately
    :return:
    """
    mpl.use('Agg', warn=False)
    f = {'family': 'Gotham', 'weight': 'light', 'size': sz}
    mpl.rc('font', **f)
    mpl.rcParams['lines.linewidth'] = 0.75

def save(fig, path, format='pdf'):
    # Fix the axes, dependent on graph type
    # Must be done after the plotting
    # Then set the values of the titles

    # Save the graph if desired, otherwise show the graph
    with SuppressErrors():
        filename = ''.join([path, '.', format])
        plt.savefig(filename, transparent=True)
    plt.close(fig)

def axheatmap(fig, ax, data, borders, tracetype='dff', cmax='auto'):
    """
    Plot a heatmap of data data with interstitial boundaries defined by borders.
    :param fig: mpl figure
    :param ax: mpl axis
    :param data: 2d matrix of data to display as heatmap with d1 vertical
    :param borders: dict of groups to vertically separate
    :param tracetype: 'dff' or 'deconvolved', used for cmap
    :param cmax: set to a float if desired, otherwise fixed value based on tracetype
    :return: heatmap on mpl axis
    """

    # Get the sizes for plotting
    ncells = np.shape(data)[0]
    nframes = np.shape(data)[1]
    clrs = pool.config.colors()

    if 'dff' in tracetype:
        if isinstance(cmax, str):
            vrange = (-0.1, 0.1)
        else:
            vrange = (-1*cmax, cmax)
        cmap = colors.gradbr()
    else:
        if isinstance(cmax, str):
            vrange = (0, 0.1)
        else:
            vrange = (0, cmax)
        cmap = plt.get_cmap('Greys')

    im = ax.pcolormesh(range(nframes), range(ncells), data, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
    im.set_rasterized(True)
    # fig.colorbar(im)

    for gr in borders:
        grname = gr.split('-')[0]
        ax.plot([0.5, nframes-0.5], [ncells - borders[gr], ncells - borders[gr]], lw=1.5, color=clrs[grname], clip_on=False)

def axxtitle(fig, ax, title):
    """
    Set the x title on a figure
    :param fig:
    :param ax:
    :param title:
    :return:
    """

    ax.set_xlabel(title)
    plt.subplots_adjust(bottom=0.20)

def axytitle(fig, ax, title):
    """
    Set the x title on a figure
    :param fig:
    :param ax:
    :param title:
    :return:
    """

    ax.set_ylabel(title)

def axtitle(fig, ax, title):
    """
    Set the y-axis title
    :param fig:
    :param ax:
    :param title:
    :return:
    """

    plt.suptitle(title, **{'family': 'Gotham', 'weight': 'book', 'size': 24, 'va': 'top', 'y': 0.995, 'color': 'black'})

def borderrange(borders, ncells):
    """
    Return the min and maxes of each border
    :param borders: single run's sort borders
    :param ncells: the number of cells in a single day (for the last border)
    :return: border ranges sorted into a list
    """

    sortbords = sorted([(borders[key], key) for key in borders if borders[key] > -1])
    ranges = [(sortbords[i][1], sortbords[i][0], sortbords[i+1][0], sortbords[i+1][0] - sortbords[i][0])
              for i in range(len(sortbords) - 1)]
    ranges.append((sortbords[-1][1], sortbords[-1][0], ncells, ncells - sortbords[-1][0]))
    return ranges

def sortorder(andb, mouse, date, analysis=''):
    """
    Return the sort order based on an analysis and date
    :param date:
    :param analysis:
    :param mouse: mouse name, required for glm labels
    :return: sort order
    """

    if 'labels' in analysis:
        # Extract the ensure and quinine groups from the labels
        groups = ['ensure-only', 'ensure-multiplexed', 'quinine-only',
                  'quinine-multiplexed']
        labels = outfns.labels(andb, mouse, date, 0.05, combine=True)
        sortcat = np.zeros(len(labels['plus']), dtype=np.int32) - 1
        for i, gr in enumerate(groups):
            sortcat[(labels[gr]) & (sortcat < 0)] = i

        # Merge in with original sorting
        extrasort = andb.get('sort-order', mouse, date)[::-1]
        branges = borderrange(andb.get('sort-borders', mouse, date), len(extrasort))
        special = np.nonzero(sortcat > -1)[0]

        newgs = ['plus', 'ensure-only', 'ensure-multiplexed', 'neutral', 'minus',
                 'quinine-only', 'quinine-multiplexed']

        borders = {}
        newsort = []
        for gr in newgs:
            borders[gr] = len(newsort)
            if gr not in groups:
                gbrange = branges[[i[0] for i in branges].index(gr)]
                srange = extrasort[gbrange[1]:gbrange[2]]
                for cell in srange:
                    if cell not in special:
                        newsort.append(cell)

            else:
                lrange = np.nonzero(labels[gr])[0]
                for cell in extrasort:
                    if cell in lrange:
                        newsort.append(cell)

        return newsort[::-1], borders

    # if analysis == '': return andb.get('sort-order', date), andb.get('sort-borders', date)
    if analysis == '':
        return andb.get('sort-simple', mouse, date), andb.get('sort-simple-borders', mouse, date)
    elif 'rot-sort-order-plus' in analysis:
        return andb.get('rot-sort-order-plus', mouse, date), andb.get('rot-sort-borders-plus', mouse, date)
    elif 'rot-sort-order-minus' in analysis:
        return andb.get('rot-sort-order-minus', mouse, date), andb.get('rot-sort-borders-minus', mouse, date)
    elif analysis == 'latency': return andb.get('sort-latency', mouse, date), andb.get('sort-latency-borders', mouse, date)

    vals = andb.get(analysis, mouse, date)
    order = sorted([(v, i) for i, v in enumerate(vals)])
    return np.array([i[1] for i in order]), {}

def reducepoints(x, y, n=2000, further=True):
    """
    Reduce the total number of x, y coordinates for plotting. The
    algorithm looks windows roughly one pixel wide and plots the
    minimum and maximum point within that window. NOTE: both the min
    and max for each n will be determined. This will yield a length
    of approximately 2n. If further, remove nonessential points.
    """

    # Can only work on blocks
    if len(x) < n*3: return (x, y)

    # Calculate the block size to average over
    block = int(math.floor(float(len(x))/n))
    newn = int(math.ceil(float(len(x))/block))
    ox, oy = np.zeros(2*newn), np.zeros(2*newn)

    # Search over each block for the min and max y value, order
    # correctly, and add to the output
    for i in range(newn):
        pmn = np.argmin(y[i*block:(i + 1)*block])
        pmx = np.argmax(y[i*block:(i + 1)*block])

        if pmn < pmx:
            ox[2*i], oy[2*i] = x[i*block + pmn], y[i*block + pmn]
            ox[2*i + 1], oy[2*i + 1] = x[i*block + pmx], y[i*block + pmx]
        else:
            ox[2*i + 1], oy[2*i + 1] = x[i*block + pmn], y[i*block + pmn]
            ox[2*i], oy[2*i] = x[i*block + pmx], y[i*block + pmx]

    if further:
        last = -1
        match = 0

        # Search through all values and set >= triplets to 0
        for i in range(len(ox)):
            if oy[i] != last:
                last = oy[i]
                match = 0
            else:
                match += 1
                if match > 1:
                    ox[i - 1] = np.nan

        # Eliminate those positions where ox is nan
        if np.sum(np.isnan(ox)) > 0:
            oy = oy[np.isfinite(ox)]
            ox = ox[np.isfinite(ox)]

    return ox, oy

# =============================================================================================== #
# Class necessary for suppressing errors

# Define a context manager to suppress stdout and stderr.
class SuppressErrors(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])