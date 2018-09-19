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


def summary_page(sorter, figsize=None, **kwargs):
    """Return a summary page of data analyzed and settings used.

    Parameters
    ----------
    sorter : DateSorter or RunSorter
        Really any iterable will work. Adds str of the object and everything it
        iterates over.
    figsize : tuple, optional
        2-element tuple of the size of the figure.
    **kwargs
        Values for any additional keyword arguments are added as text.

    Notes
    -----
    There is no attempt to actual estimate the size of the text, it's just
    assumed to fit on the page. If the text is too long, it will go off the
    bottom. Could add fontsize argument if needed.

    """
    if figsize is None:
        figsize = (8.5, 11)
    fig = plt.figure(figsize=figsize)

    header = 'Summary sheet - {time} - {sha1}\n\n'.format(
        time=time.asctime(), sha1=git_revision())

    divider = '-------------------------------------------------------------\n'

    sorter_text = str(sorter) + '\n'
    for x in sorter:
        sorter_text += ' ' + str(x) + '\n'

    kwargs_text = pprint.pformat(kwargs)

    fig_text = header + 'Experiments\n' + divider + sorter_text + \
        '\nParameters\n' + divider + kwargs_text

    fig.text(0.05, 0.97, fig_text, va='top', ha='left', fontsize=5)

    return fig


def save_figs(save_path, figs):
    """Save figs to a file.

    'figs' can be a generator and they will be consumed and closed 1 by 1.
    Uses a temporary file so that a crash in the middle doesn't corrupt the
    destination on an overwrite.

    """
    mkdir_p(os.path.dirname(save_path))
    temp_save_path = save_path + '.tmp'
    # noinspection PyBroadException
    try:
        if save_path.endswith('.pdf'):
            from matplotlib.backends.backend_pdf import PdfPages
            pp = PdfPages(temp_save_path)
            for fig in figs:
                pp.savefig(fig)
                plt.close(fig)
            pp.close()
    except:
        # os.remove(temp_save_path)
        raise
    else:
        os.rename(temp_save_path, save_path)
    finally:
        try:
            os.remove(temp_save_path)
        except OSError:
            pass


def default_parser(arguments=('mouse', 'date', 'tags'), **kwargs):
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

    if 'mice' in arguments:
        parser.add_argument(
            '-m', '--mice', type=str, action='store', nargs='*', default=None,
            help='Mice to analyze.')
    if 'dates' in arguments:
        parser.add_argument(
            '-d', '--dates', type=int, action='store', nargs='*', default=None,
            help='Dates to analyze.')
    if 'runs' in arguments:
        parser.add_argument(
            '-r', '--runs', type=int, action='store', nargs='*', default=None,
            help='Runs to analyze.')
    if 'tags' in arguments:
        parser.add_argument(
            '-t', '--tags', type=str, action='store', nargs='*', default=None,
            help='Additional tags to filter mouse/date/run on.')
    if 'overwrite' in arguments:
        parser.add_argument(
            '-o', '--overwrite', action='store_true',
            help='If True, overwrite pre-existing files.')

    return parser


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
