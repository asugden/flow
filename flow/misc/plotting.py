"""General plotting functions."""
import numpy as np
import seaborn as sns


def layout_subplots(n_plots, height=11., width=8.5, **kwargs):
    """Layout subplots to fit square axes on a fixed size figure.

    Determines the optimal number of rows and columns to fit the given number
    of square axes on a page.

    Parameters
    ----------
    n_plots : int
        Number of plots to fit.
    height, width : float
        Page dimensions, in inches.

    Other parameters
    ----------------
    **kwargs
        Everything else is passed to plt.subplots

    """
    rows = 0.
    cols = 0.
    p = 0
    while p < n_plots:
        rows += 1.
        cols = int(width / (height / rows))
        p = rows * cols

    fig, axs = plt.subplots(
        int(rows), int(cols), figsize=(width, height), squeeze=False, **kwargs)

    for ax in axs.flatten()[n_plots:]:
        ax.set_visible(False)

    return fig, axs


def plot_traces(ax, traces, t_range):
    """Plot a series of traces staggered along the y-axis.

    Parameters
    ----------
    ax : mpl.axes
    traces : ndarray
        (time x ntraces)
    t_range : 2-element tuple
        minimum and maximum value

    """
    x_range = np.linspace(t_range[0], t_range[1], traces.shape[0])

    spacing = np.nanmean(traces) + 2*np.nanstd(traces)
    steps = np.arange(traces.shape[1]) * spacing
    stepped_traces = traces + steps
    ax.plot(x_range, stepped_traces)

    sns.despine(ax=ax)
    if t_range[0] < 0 and t_range[1] > 0:
        ticks = [t_range[0], 0, t_range[1]]
        ax.axvline(0, color='k', linestyle='--')
    else:
        ticks = t_range
    ax.set_xticks(ticks)
    ax.set_ylim(np.nanmin(stepped_traces) - spacing,
                np.nanmax(stepped_traces) + spacing)
    ax.set_yticks([steps[0], steps[-1]])
    ax.set_yticklabels([1, traces.shape[1]])
    ax.set_ylabel('Trial')
    ax.set_xlabel('Time (s)')
