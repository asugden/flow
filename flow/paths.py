import os
import os.path as opath
import numpy as np
from scipy.io import loadmat

from .misc import wordhash
from .trace2p import Trace2P
from . import config

params = config.params()

datad = params['paths'].get('data', '/data')
outd = params['paths'].get('output', '/output')
graphd = params['paths'].get('graph', '/graphs')


def extract_run_specific_pars(pars):
    extracted = ['mouse', 'training-date', 'comparison-date', 'comparison-run']
    out = {}
    for p in pars:
        if p not in extracted:
            out[p] = pars[p]
    return out

def dataframe(name):
    """Save a pandas datafarme to a directory in output

    :param name: name of file, to which h5 will be appended if not there.
    :return: path
    """

    if name[-3:] != '.h5':
        name = name + '.h5'

    path = opath.join(outd, 'dataframes')
    if not opath.exists(path):
        os.mkdir(path)

    path = opath.join(path, name)
    return path

def gett2p(mouse, date, run):
    path = opath.join(datad, '%s/%s/%s_%s_%03i.simpcell' % (
        mouse, date, mouse, date, run))
    out = Trace2P(path)
    return out

def getonsets(mouse, date=None, run=None):
    """
    Return the extra onsets file
    :param mouse: can be mouse name, str, or the complete filename if date and run are left none
    :param date: date, str, or None if filename
    :param run: run, int, or None if filename
    :return: loaded onsets file if possible
    """

    if date is None or run is None:
        path = opath.join(datad, 'onsets/%s' % (mouse))
    else:
        path = opath.join(datad, 'onsets/%s_%s_%03i.onsets' % (mouse, date, run))

    if not opath.exists(path):
        return None
    else:
        out = loadmat(path)
        return out

def getglm(mouse, date):
    path = opath.join(datad, '%s/%s/%s_%s.simpglm' % (mouse, date, mouse, date))
    if not opath.exists(path):
        return None
    else:
        data = loadmat(path, appendmat=False)
        cgs = [str(data['cellgroups'].flatten()[i][0]) for i in range(len(data['cellgroups'].flatten()))]
        devexp = data['deviance_explained']
        return [cgs, devexp, data]


def cosdists():
    return '%s/cosdists.mat' % outd


def glmpath(mouse, date):
    """
    Get the path to a .simpglm file, set to None if it does not exist
    :param mouse: mouse name, str
    :param date: date, str
    :return:
    """

    path = opath.join(datad, '%s/%s/%s_%s.simpglm' % (mouse, date, mouse, date))
    if not opath.exists(path):
        return None
    else:
        return path

def gettclassmarginals(pars):
    """
    Return the path to the marginal probabilities measured from the time classifier.
    :return: path, str
    """
    word = wordhash.word(extract_run_specific_pars(pars))
    path = opath.join(outd, 'time-classifier-training')
    if not opath.exists(path): os.mkdir(path)
    return opath.join(path, '%s-time-classifier-marginals.mat'%word)

def exist(mouse, date, run):
    path = opath.join(datad, '%s/%s/%s_%s_%03i.simpcell' % (mouse, date, mouse, date, run))
    return opath.isfile(path)

def ids(mouse, date):
    """
    Return the crossday cell IDs if they exist, otherwise return an
    empty path.
    """

    path = opath.join(datad, '%s/%s/%s_%s_crossday-cell-ids.txt' % (mouse, date, mouse, date))
    if opath.isfile(path): return path
    else: return ''

def pairids(mouse, day1, day2):
    """
    Return the paired ids of cells
    """

    mpath = opath.join(datad, '%s/crossday' % (mouse))
    if not opath.exists(mpath): return [], []

    day1, day2 = str(day1), str(day2)

    fs = os.listdir(mpath)
    for name in fs:
        cd = name.split('--')
        if len(cd) == 2:
            cd = cd[0].split('-')
            if len(cd) == 2:
                if day1 in cd[0] and day2 in cd[1]:
                    x, y = np.loadtxt(opath.join(mpath, name), unpack=True)
                    x = x.astype(np.int32) - 1
                    y = y.astype(np.int32) - 1
                    return x, y
                elif day2 in cd[0] and day1 in cd[1]:
                    y, x = np.loadtxt(opath.join(mpath, name), unpack=True)
                    x = x.astype(np.int32) - 1
                    y = y.astype(np.int32) - 1
                    return x, y

    return [], []

def db(mouse, old=False):
    """
    Return the path to the analysis database per mouse.
    """

    path = outd
    if old:
        path = opath.join(path, 'old-standard')

    path = opath.join(path, mouse)
    if not opath.exists(path): os.mkdir(path)

    path = opath.join(path, '%s.db'%mouse)
    #if opath.isfile(path): return path
    #else: return ''
    return path

def udb(mouse, old=False):
    """
    Return the path to the analysis database per mouse.
    """

    path = outd
    if old:
        path = opath.join(path, 'old-standard')

    path = opath.join(path, '%s/%s-updated.db' % (mouse, mouse))
    #if opath.isfile(path): return path
    #else: return ''
    return path

def graph(pars):
    word = wordhash.word(extract_run_specific_pars(pars))
    mouse = pars['mouse']
    date = pars['training-date']
    crun = pars['comparison-run']

    # Base/mouse
    path = opath.join(graphd, mouse)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date
    path = opath.join(path, date)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date/run-parameterWord
    path = opath.join(path, '%03i-%s' % (crun, word))
    if not opath.exists(path): os.mkdir(path)

    return path

def graphcrossday():
    # Base/crossday
    path = opath.join(graphd, 'crossday')
    if not opath.exists(path): os.mkdir(path)

    return opath.join(path, '')

def graphgroup(pars={}, group='', classifier=True):
    # Base/plots/classifier-keyword/group/

    path = opath.join(graphd, 'plot')
    if not opath.exists(path): os.mkdir(path)

    if pars != {} and classifier:
        word = wordhash.word(extract_run_specific_pars(pars))
        path = opath.join(path, word)
        if not opath.exists(path): os.mkdir(path)

    if len(group) > 0:
        path = opath.join(path, group)
        if not opath.exists(path): os.mkdir(path)

    return opath.join(path, '')

def graphmdr(pars):
    word = wordhash.word(extract_run_specific_pars(pars))
    mouse = pars['mouse']
    date = pars['training-date']
    crun = pars['comparison-run']

    # Base/mouse
    path = opath.join(graphd, mouse)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date
    path = opath.join(path, date)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date/run-parameterWord
    path = opath.join(path, '%03i-%s' % (crun, word))
    if not opath.exists(path): os.mkdir(path)

    # Add mouse, date, and run to the beginning of the graph title
    path = opath.join(path, '%s-%s-%02i' % (mouse, date, crun))

    return path

def classifierword(pars):
    """
    Return a random word generated from the hash of a classifier
    :param pars: parameters, from settings
    :return: word, str
    """

    return wordhash.word(extract_run_specific_pars(pars))

def output(pars):
    word = wordhash.word(extract_run_specific_pars(pars))
    # print 'Classifier %s' % (word)
    mouse = pars['mouse']
    date = pars['training-date']
    crun = pars['comparison-run']

    # Base/mouse
    path = opath.join(outd, mouse)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date
    path = opath.join(path, date)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date/run-parameterWord
    path = opath.join(path, '%03i-%s' % (crun, word))
    if not opath.exists(path): os.mkdir(path)

    return path

def neuralnet(mouse, date, netpars={}):
    word = wordhash.word(extract_run_specific_pars(netpars))
    # print 'Classifier %s' % (word)

    # Base/mouse
    path = opath.join(outd, mouse)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date
    path = opath.join(path, date)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date/nn-parameterWord.h5
    path = opath.join(path, 'nn-%s.h5' % word)

    return path

def training(pars):
    word = wordhash.word(extract_run_specific_pars(pars))
    # print 'Classifier %s' % (word)
    mouse = pars['mouse']
    date = pars['training-date']

    # Base/mouse
    path = opath.join(outd, mouse)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date
    path = opath.join(path, date)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date/run-parameterWord
    path = opath.join(path, '%s-training.mat' % word)

    return path

def ctraindump(pars):
    word = wordhash.word(extract_run_specific_pars(pars))
    # print 'Classifier %s' % (word)
    mouse = pars['mouse']
    date = pars['training-date']

    # Base/mouse
    path = opath.join(outd, mouse)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date
    path = opath.join(path, date)
    if not opath.exists(path): os.mkdir(path)
    # Base/mouse/date/parameterWord
    path = opath.join(path, '%s-ctrain.pickle'%word)

    return path

def pupilpos(mouse, date, run):
    """
    Get the ancillary pupil position
    :param mouse: mouse name, str
    :param date: date, str (yymmdd)
    :param run: run, int
    :return:
    """

    path = opath.join(datad, 'pupilpos')
    path = opath.join(path, '%s-%s-%03i-pupil.mat' % (mouse, date, run))
    if opath.exists(path):
        return loadmat(path)
    else:
        return None

def xlabel(mouse, date):
    """
    Crossday labels
    :param mouse:
    :param date:
    :return:
    """

    # Base/mouse
    path = opath.join(outd, mouse)
    if not opath.exists(path): os.mkdir(path)

    path = opath.join(path, 'xlabel')
    if not opath.exists(path): os.mkdir(path)

    # Base/mouse/xlabel/date.txt
    path = opath.join(path, '%s.txt' % (str(date)))
    return path