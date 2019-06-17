"""Miscellaneous helper functions."""
from builtins import str
from past.builtins import basestring

import argparse
import collections
import datetime
import errno
from getpass import getuser
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pprint
import scipy.io as spio
from scipy.sparse import issparse
import subprocess
import time

from . import wordhash


def timestamp():
    """Return the current time as a timestamp string."""
    return datetime.datetime.strftime(
        datetime.datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')


def datestamp(compact=False):
    """Return the current date as a timestamp string."""
    if compact:
        return datetime.datetime.strftime(
            datetime.datetime.now(), '%y%m%d')
    else:
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
        else:
            assert len(figs) == 1
            fig.savefig(temp_save_path)
            plt.close(fig)
    except:
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
    elif 'mouse' in arguments:
        parser.add_argument(
            '-m', '--mouse', type=str, action='store',
            help='Mouse to analyze.')

    if 'dates' in arguments:
        parser.add_argument(
            '-d', '--dates', type=int, action='store', nargs='*', default=None,
            help='Dates to analyze.')
    elif 'date' in arguments:
        parser.add_argument(
            '-d', '--date', type=int, action='store',
            help='Date to analyze.')

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
    if 'verbose' in arguments:
        parser.add_argument(
            '-v', '--verbose', action='store_true',
            help='Be verbose.')

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
                    not issparse(data[key]) and \
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


def savemat(filename, data, appendmat=False):
    """Save a dictionary to a mat file."""
    # Eventually this should cleanup the input dict, specifically making sure
    # that the py2/3 newstr is converted to an actual strings since
    # spio.savemat doesn't know how to handle them.
    spio.savemat(filename, data, appendmat=appendmat)


def parse_date(datestr):
    """Parse our standard format date string into a datetime object.

    Parameters
    ----------
    datestr : str or int
        String or int to be parsed.

    Returns
    -------
    datetime

    """
    datestr = str(datestr)
    try:
        date = datetime.datetime.strptime(datestr, '%y%m%d')
    except ValueError:
        raise ValueError(
            'Malformed date. Should be in YYMMDD format, e.g. 180723')
    return date


def notebook_word():
    """Converts a Jupyter notebook runtime connection file to a simple word.

    Must be called within a Jupyter notebook. Combined with 'notebook_file',
    these functions allow you to connect to an active Jupyter notebook at the
    command line.

    notebook_file(notebook_word()) will return back the name of the connection
    file.

    Returns
    -------
    word : str

    """
    from ipykernel import get_connection_file

    return wordhash.word(str(os.path.basename(get_connection_file())))


def notebook_file(word, path=None):
    """Converts a word from 'notebook_word' back to a filename.

    Used to connect to an active Jupyter notebook from the command line.

    To get matching file:
        $> python -c "import jzap.misc; print(jzap.misc.notebook_file('WORD'))"
    To connect to an existing notebook:
        $> jupyter console --existing FILE

    Arguments
    ---------
    word : str
        A simple word computing by 'notebook_word'.
    path : str, optional
        Specify the location of the jupyter notebook runtime connection files.
        If None, attempts to guess location.

    Returns
    -------
    file : str

    """
    if path is None:
        paths = ['/home/jupyter/.local/share/jupyter/runtime',
                 '/home/{}/.local/share/jupyter/runtime'.format(getuser())]
    else:
        paths = [path]
    for p in paths:
        if os.path.isdir(p):
            for file in os.listdir(p):
                if wordhash.word(str(file)) is word:
                    return file
    return ''


def md5(fname):
    """Calculate the md5 hash of a file.

    Copied from Stack Overflow:
    https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file

    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def matlabifypars(pars):
    """Convert the parameters into a Matlab-readable version."""

    out = {}
    for p in pars:
        mlname = p[:31].replace('-', '_')
        if isinstance(pars[p], dict):
            out[mlname] = matlabifypars(pars[p])
        elif pars[p] is None:
            out[mlname] = np.nan
        else:
            out[mlname] = pars[p]
    return out
