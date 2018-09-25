import numpy as np
from scipy.signal import gaussian
import scipy.io as spio

from . import metadata as metadata
from . import paths


def glm(mouse, date):
    """
    Return a new instance of a GLM
    :param mouse: mouse name, str
    :param date: date, str
    :return: instance of class GLM
    """

    out = GLM(mouse, date)
    if not out.exists:
        return False
    else:
        return out

def labels(mouse, date, minpred=0.01, minfrac=0.05):
    """
    Return the GLM labels for a particular mouse and date
    :param mouse: mouse name, str
    :param date: date, str
    :param minpred: the minimum variance predicted by all glm filters
    :param minfrac: the minimum fraction of glm variance explained by filters for each cell type
    :return: a dict of the unit vectors for each group
    """

    out = GLM(mouse, date)
    if not out.exists:
        return False
    else:
        return out.labels(minpred, minfrac)

def unitvectors(mouse, date, trange=(0, 2), rectify=True, hz=None):
    """
    Get expected values across a time range given GLM coefficients.
    :param mouse: mouse name, str
    :param date: date, str
    :param trange: time range in seconds, tuple
    :param rectify: set values < 0 equal to 0 if True
    :param hz: the frequency of the recording, will automatically find if necessary
    :return: a dict of the unit vectors for each group
    """

    out = GLM(mouse, date)
    if not out.exists:
        return False
    else:
        return out.vectors(trange, rectify, hz)


class GLM:
    """
    A class that interfaces with a .simpglm file
    """

    def __init__(self, mouse, date):
        """
        Create a new GLM instance with a given mouse and date
        :param mouse: mouse name, str
        :param date: date, str
        """

        self.mouse = mouse
        self.date = date
        self.exists = False
        self.freq = None
        path = paths.glmpath(mouse, date)

        if path is not None:
            self.d = loadmatpy(path)

            if 'behaviornames' in self.d:
                self.exists = True

                self.names = [str(v) for v in self.d['behaviornames']]
                self.lags = self.d['behaviorlags'].flatten()
                self.coeffs = self.d['coefficients']
                self.pars = self.d['pars']
                self.cellgroups = [str(v) for v in self.d['cellgroups']]
                self.devexp = self.d['deviance_explained']

    def groups(self, short=False):
        """
        Return a list of all cellgroups
        :param short: set to True if only simple cell groups should be returned
        :return: list of cell groups, str
        """

        if short:
            return self.cellgroups
        else:
            return self.names

    def basis(self, group, trange=(0, 2), hz=None):
        """
        Reconstruct a basis function
        :param group: the group name over which to reconstruct the basis function
        :param trange: time range to include
        :param hz: frequency of the recording, will automatically find if necessary
        :return: a matrix of ncells x ntimes with basis vectors relative to stimulus
        """

        hz = self._frequency(hz)

        ncells = np.shape(self.coeffs)[0]
        ncoeffs = np.shape(self.coeffs)[1]
        grnames = np.arange(ncoeffs)[np.array([name == group for name in self.names])]
        sigma = self.pars['gaussian_s']*4.6  # would be 2.4 if downsampled

        textreme = max(np.abs(trange[0]), np.abs(trange[1])) + self.pars['gaussian_s']*10
        textreme = int(round(textreme))
        ts = np.arange(-textreme, textreme, 1.0/hz)
        recon = np.zeros((ncells, len(ts)))

        for gr in grnames:
            fr = (np.abs(ts - self.lags[gr])).argmin()
            recon[:, fr] = self.coeffs[:, gr]

        nframes = int(round(self.pars['gaussian_s']*1.5*hz/2))
        norm = gaussian(2*nframes + 1, sigma)
        norm /= np.sum(norm)
        for c in range(ncells):
            recon[c, :] = np.convolve(recon[c, :], norm, mode='same')

        frange = [(np.abs(ts - trange[0])).argmin(), (np.abs(ts - trange[1])).argmin()]
        return recon[:, frange[0]:frange[1]]

    def vectors(self, trange=(0, 2), rectify=True, hz=None):
        """
        Get expected values across a time range given GLM coefficients.
        :param trange: time range in seconds, tuple
        :param rectify: set values < 0 equal to 0 if True
        :param hz: the frequency of the recording, will automatically find if necessary
        :return: a dict of the unit vectors for each group
        """

        out = {}
        for group in self.groups():
            unit = self.basis(group, trange, hz)  # ncells x ntimes
            if rectify:
                unit[unit < 0] = 0

            unit = np.nanmean(unit, axis=1)
            if np.nansum(unit) == 0:
                unit[:] = np.nan
            else:
                out[group] = unit/np.nansum(unit)

        return out

    def meanresp(self, trange=(0, 2), rectify=False, hz=None):
        """
        Get the mean responses to each of the groups
        :param trange: time range in seconds
        :param rectify: set values < 0 equal to 0 if True
        :param hz: frequency of the recording, will automatically find if necessary
        :return: a dict of the unit vectors for each group
        """

        out = {}
        for group in self.groups():
            vec = self.basis(group, trange, hz)  # ncells x ntimes
            if rectify:
                vec[vec < 0] = 0
            out[group] = vec

        return out

    def labels(self, minpred=0.01, minfrac=0.05):
        """
        Label cells by their GLM filter responses
        :param minpred: the minimum variance predicted by all glm filters
        :param minfrac: the minimum fraction of glm variance explained by filters for each cell type
        :return: dict of labels
        """

        # Get cell groups and the sufficiency of each deviance explained
        groupdev = (self.devexp[:, 1:].T*(self.devexp[1][:, 0] >= minpred)).T
        groupdev = groupdev >= minfrac
        ncells = np.shape(self.coeffs)[0]

        odict = {}
        for i, name in enumerate(self.cellgroups):
            odict[name] = groupdev[:, i]

        # Multiplexed cells should respond not just to plus correct and plus error, but also to other types
        # Similarly, licking and ensure cannot be differentiated the way we're measuring them, so they can
        # also be combined.

        multiplexed = np.sum(groupdev, axis=1)
        odict['undefined'] = multiplexed == 0
        odict['multiplexed'] = multiplexed > 1

        categories = ['plus', 'neutral', 'minus', 'ensure', 'quinine']
        for cat in categories:
            odict['%s-only'%cat] = np.bitwise_and(odict[cat], np.invert(odict['multiplexed']))
            odict['%s-multiplexed'%cat] = np.bitwise_and(odict[cat], odict['multiplexed'])

        if 'predict' in odict:
            odict['reward'] = np.bitwise_or(odict['ensure'], odict['predict'])

        return odict

    # Local functions
    def _frequency(self, freq=None):
        """
        Get the frequency if it doesn't exist
        :return: frequency, float
        """

        if freq is not None:
            self.freq = freq
        elif self.freq is None:
            runs = metadata.runs(self.mouse, self.date)
            t2p = paths.gett2p(self.mouse, self.date, runs[0])
            self.freq = t2p.framerate

        return self.freq


def loadmatpy(filename):
    """
    A modified loadmat that can account for structs as dicts.
    """

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True, appendmat=False)
    for key in data:
        if isinstance(data[key], spio.matlab.mio5_params.mat_struct):
            data[key] = _mattodict(data[key])
    return data

def _mattodict(matobj):
    """
    Recursively convert matobjs into dicts.
    :param matobj: matlab object from _check_keys
    :return: dict
    """

    out = {}
    for strg in matobj._fieldnames:
        el = matobj.__dict__[strg]
        if isinstance(el, spio.matlab.mio5_params.mat_struct):
            out[strg] = _mattodict(el)
        else:
            out[strg] = el
    return out
