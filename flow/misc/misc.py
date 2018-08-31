"""Miscellaneous helper functions."""
import argparse
import collections
import datetime
import errno
import matplotlib.pyplot as plt
import os
import pprint
import scipy.io as spio
import subprocess
import time


def timestamp():
    """Return the current time as a timestamp string."""
    return datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')


def datestamp():
    """Return the current date as a timestamp string."""
    return datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d')


def mkdir_p(path):
    """Make a directory and all missing directories along the way.

    Does not fail if the directory already exists.

    See:
    https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python

    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def git_revision():
    """Return the git SHA1 revision number of the current code."""
    path = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    sha = subprocess.check_output("git rev-parse HEAD", shell=True)
    os.chdir(path)
    return sha.strip()


def summary_page(days, lpars):
    """Return a summary page of data analyzed and settings used."""
    fig = plt.figure(figsize=(8.5, 11))

    header = 'Summary sheet - {time} - {sha1}\n\n'.format(
        time=time.asctime(), sha1=git_revision())

    divider = '-------------------------------------------------------------\n'

    days_text = ''
    days.reset()
    while days.next():
        md, _ = days.get()
        days_text += str(md) + '\n'
    days.reset()

    lpars_text = pprint.pformat(lpars)

    fig_text = header + 'Experiments\n' + divider + days_text + \
        '\nParameters\n' + divider + lpars_text

    fig.text(0.05, 0.97, fig_text, va='top', ha='left', fontsize=5)

    return fig


def save_figs(save_path, figs):
    """Save figs to a file."""
    mkdir_p(os.path.dirname(save_path))
    if save_path.endswith('.pdf'):
        from matplotlib.backends.backend_pdf import PdfPages
        pp = PdfPages(save_path)
        for fig in figs:
            pp.savefig(fig)
            plt.close(fig)
        pp.close()


def smart_parser(**kwargs):
    """Return a smart parser that will parse all arguments."""
    epilog = "The smart_parser will additionally accept any recognized" + \
        " args and add them to the resulting Namespace."
    parser = argparse.ArgumentParser(epilog=epilog, **kwargs)

    def smart_parse_args():
        args, extras = parser.parse_known_args()
        idx = 0
        while idx < len(extras):
            if extras[idx].startswith('-'):
                skip = 1
                if extras[idx].startswith('--'):
                    skip = 2
                key = extras[idx][skip:]
                idx += 1
                values = []
                while idx < len(extras) and not extras[idx].startswith('-'):
                    values.append(extras[idx])
                    idx += 1
                if not len(values):
                    raise ValueError(
                        "All unknown arguments must have at least 1 value.")
                elif len(values) == 1:
                    vars(args)[key] = values[0]
                else:
                    vars(args)[key] = values
            else:
                idx += 1
        return args

    parser.parse_args = smart_parse_args

    return parser


def default_parser(arguments=('mouse', 'date'), **kwargs):
    """Return a default parser that includes default arguments.

    Parameters
    ----------
    arguments : list of str
        List of default parameters to include.
    **kwargs
        Additional keyword arguments to pass to ArgumentParser.

    Returns
    -------
    argparse.ArgumentParser

    """
    parser = argparse.ArgumentParser(**kwargs)

    if 'mouse' in arguments:
        parser.add_argument(
            '-m', '--mouse', type=str, action='store',
            help='Mouse to analyze.')
    if 'date' in arguments:
        parser.add_argument(
            '-d', '--date', type=int, action='store',
            help='Date to analyze.')

    return parser


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
    p = 0
    while p < n_plots:
        rows += 1.
        cols = int(width / (height / rows))
        p = rows * cols

    fig, axs = plt.subplots(
        int(rows), int(cols), figsize=(width, height), **kwargs)

    for ax in axs.flatten()[n_plots:]:
        ax.set_visible(False)

    return fig, axs


def loadmat(filename):
    """Load mat files and convert structs to dicts.

    Originally implemented by Francisco Luongo, see:
    https://scanbox.org/2016/09/02/reading-scanbox-files-in-python/
    Based in part onL
    http://stackoverflow.com/questions/7008608/
    scipy-io-loadmat-nested-structures-i-e-dictionaries

    """
    def check_keys(data):
        """Check if entries in dictionary are mat-objects.

        If yes, todict is called to change them to nested dictionaries

        """
        for key in data:
            if isinstance(data[key], dict):
                data[key] = check_keys(data[key])
            elif isinstance(data[key], spio.matlab.mio5_params.mat_struct):
                data[key] = todict(data[key])
            elif isinstance(data[key], collections.Iterable) and \
                    not isinstance(data[key], basestring) and \
                    len(data[key]) and \
                    isinstance(data[key][0],
                               spio.matlab.mio5_params.mat_struct):
                data[key] = [todict(item) for item in data[key]]
        return data

    def todict(matobj):
        """Construct nested dictionaries from matobjects."""
        data = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                data[strg] = todict(elem)
            else:
                data[strg] = elem
        return check_keys(data)

    data = spio.loadmat(
        filename, struct_as_record=False, squeeze_me=True, appendmat=False)
    return check_keys(data)
