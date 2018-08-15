"""Miscellaneous helper functions."""
import argparse
import datetime
import errno
import matplotlib.pyplot as plt
import os
import pprint
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
