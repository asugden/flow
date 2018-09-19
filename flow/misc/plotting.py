"""General plotting functions."""
from copy import copy
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


def right_label(ax, label, **kwargs):
    """Add a label to the right of an axes"""
    Bbox = ax.figbox
    ax.figure.text(
        Bbox.p1[0] + 0.02, (Bbox.p1[1] + Bbox.p0[1]) / 2, label, **kwargs)


def plot_traces(ax, traces, t_range, normalize=False, errors=None):
    """Plot a series of traces staggered along the y-axis.

    Parameters
    ----------
    ax : mpl.axes
    traces : ndarray
        (time x ntraces)
    t_range : 2-element tuple
        minimum and maximum value
    normalize : boolean
        If True, z-score all traces.
    errors : list of boolean, optional
        If not None, len(errors) must equal traces.shape[1]. True where the
        trials were incorrect.

    """
    if normalize:
        traces = (traces - np.nanmean(traces, 0)) / np.nanstd(traces, 0)

    x_range = np.linspace(t_range[0], t_range[1], traces.shape[0])

    # Infer some reasonable spacing
    spacing = np.nanmedian(traces) + np.nanstd(traces)
    steps = np.arange(traces.shape[1]) * spacing
    stepped_traces = traces + steps
    with sns.color_palette():
        ax.plot(x_range, stepped_traces)

    # Add the mean response at the top
    trace_mean_step = steps[-1] + 3*spacing
    trace_mean = np.nanmean(traces, 1) + trace_mean_step
    trace_5 = np.nanpercentile(traces, 2.5, axis=1) + trace_mean_step
    trace_95 = np.nanpercentile(traces, 97.5, axis=1) + trace_mean_step
    ax.plot(x_range, trace_mean, color='k')
    ax.fill_between(
        x_range, trace_5, trace_95, color='k', alpha=0.2)

    # Figure out x-ticks
    sns.despine(ax=ax)
    if t_range[0] < 0 and t_range[1] > 0:
        ticks = [t_range[0], 0, t_range[1]]
        ax.axvline(0, color='k', linestyle='--')
    else:
        ticks = t_range
    tick_labels = copy(ticks)

    # Add error marks if present
    if errors is not None:
        # Guess a reasonable spacing
        t_gap = (t_range[1] - t_range[0]) * 0.05
        error_x_min = t_range[1] + t_gap
        error_x_max = t_range[1] + 2*t_gap
        good_trials = [step for step, error in zip(steps, errors) if not error]
        bad_trials = [step for step, error in zip(steps, errors) if error]
        for t in good_trials:
            ax.plot([error_x_min, error_x_max], [t, t], color='g', lw=3)
        for t in bad_trials:
            ax.plot([error_x_min, error_x_max], [t, t], color='r', lw=3)
        ticks.append(np.mean([error_x_min, error_x_max]))
        tick_labels.append('err')

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(np.nanmin(stepped_traces) - spacing,
                np.nanmax(trace_95) + spacing)
    ax.set_yticks([steps[0], steps[-1], steps[-1] + 3*spacing])
    ax.set_yticklabels([1, traces.shape[1], r'mean $\pm$ 90%'])
    ax.set_ylabel('Trial')
    ax.set_xlabel('Time (s)')
