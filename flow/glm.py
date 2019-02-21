"""Interface for simpglm files."""
from __future__ import division
from builtins import str
from builtins import range
from builtins import object

try:
    from bottleneck import nanmean, nansum
except ImportError:
    from numpy import nanmean, nansum
from copy import deepcopy
import numpy as np
from scipy import optimize
from scipy.signal import gaussian
import warnings

from . import misc
from . import paths


def glm(mouse, date, hz=None):
    """
    Return a new instance of a GLM.

    :param mouse: mouse name, str
    :param date: date, str
    :return: instance of class GLM
    """

    out = GLM(mouse, date, hz=hz)
    if not out.exists:
        return False
    else:
        return out


def labels(mouse, date, minpred=0.01, minfrac=0.05):
    """
    Return the GLM labels for a particular mouse and date.

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

    out = GLM(mouse, date, hz=hz)
    if not out.exists:
        return False
    else:
        return out.vectors(trange, rectify)


class GLM(object):
    """A class that interfaces with a .simpglm file."""

    def __init__(self, mouse, date, hz=None):
        """
        Create a new GLM instance with a given mouse and date.

        :param mouse: mouse name, str
        :param date: date, str

        """
        self.mouse = mouse
        self.date = date
        if hz is None:
            self.hz = 15.49
        else:
            self.hz = hz
        self.exists = False
        self.freq = None

        self._original_coeffs = None
        self._original_devexp = None

        path = paths.glmpath(mouse, date)

        if path is not None:
            self.d = misc.loadmat(path)

            if 'behaviornames' in self.d:
                self.exists = True

                self.names = [str(v) for v in self.d['behaviornames']]
                self.lags = self.d['behaviorlags'].flatten()
                self.coeffs = self.d['coefficients']
                self.pars = self.d['pars']
                self.cellgroups = [str(v) for v in self.d['cellgroups']]
                self.devexp = self.d['deviance_explained']

    def __repr__(self):
        return "GLM(mouse={}, date={})".format(self.mouse, self.date)

    def groups(self, short=False):
        """
        Return a list of all cellgroups.

        :param short: set to True if only simple cell groups should be returned
        :return: list of cell groups, str
        """

        if short:
            return self.cellgroups
        else:
            return self.names

    def subset(self, vector):
        """
        Reorder cells and select a subset, used for matching across days.

        Parameters
        ----------
        vector : numpy array
            A vector with which cell traces should be reordered. If None, return to default

        """

        if self._original_coeffs is not None:
            self.coeffs = deepcopy(self._original_coeffs)
            self.devexp = deepcopy(self._original_devexp)

        if vector is not None:
            if self._original_coeffs is None:
                self._original_coeffs = deepcopy(self.coeffs)
                self._original_devexp = deepcopy(self.devexp)

            self.coeffs = self.coeffs[vector, :]
            self.devexp = self.devexp[vector, :]

    def basis(self, group, trange=(0, 2), hz=None):
        """
        Reconstruct a basis function.

        :param group: the group name over which to reconstruct the basis function
        :param trange: time range to include
        :param hz: the frequency of the recording, will automatically find if necessary
        :return: a matrix of ncells x ntimes with basis vectors relative to stimulus
        """
        if hz is None:
            hz = self.hz
        else:
            # 190207
            warnings.warn(
                "Pass framerate argument to the GLM init.",
                DeprecationWarning)
        if group not in set(self.groups()):
            raise ValueError('Group not found: {}'.format(group))
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
        if hz is not None:
            # 190207
            warnings.warn(
                "Pass framerate argument to the GLM init.",
                DeprecationWarning)
        out = {}
        for group in set(self.groups()):
            unit = self.basis(group, trange, hz=hz)  # ncells x ntimes
            if rectify:
                unit[unit < 0] = 0

            unit = np.nanmean(unit, axis=1)
            if np.nansum(unit) == 0:
                unit[:] = np.nan
            else:
                out[group] = unit/np.nansum(unit)

        return out

    def protovector(
            self, group, trange=(0, 1), rectify=False, err=-1,
            remove_group=None):
        """
        Get the prototypical vector of a GLM group.

        Parameters
        ----------
        group : str
            Group name from glm.groups().
        trange : 2-element tuple of float
            Time range in seconds.
        rectify : bool
            If True, set all negative components to 0.
        err : {-1, 0, 1}
            Determined handling of trial errors for visual groups. -1 for all
            trials, 0 for correct trials, 1 for incorrect trials.
        remove_group : str, optional
            If not None, remove the given group response from the protovector.

        Returns
        -------
        np.ndarray

        """
        if group in ['plus', 'neutral', 'minus']:
            if err == -1:
                correct = self.protovector(
                    group + '_correct', trange=trange, rectify=rectify)
                try:
                    miss = self.protovector(
                        group + '_miss', trange=trange, rectify=rectify)
                except ValueError:
                    return correct
                else:
                    return (correct + miss) / 2.
            elif err == 0:
                return self.protovector(
                    group + '_correct', trange=trange, rectify=rectify)
            elif err == 1:
                return self.protovector(
                    group + '_miss', trange=trange, rectify=rectify)
            else:
                raise ValueError(
                    'Unrecognized err argument, must be in {-1, 0, 1}')

        # From GLM.vectors()
        unit = self.basis(group, trange)
        if rectify:
            unit[unit < 0] = 0
        unit = nanmean(unit, axis=1)
        if nansum(unit) == 0:
            unit.fill(np.nan)
        else:
            unit /= nansum(unit)

        # From outfns._remove_visual_components
        if remove_group is not None:
            def fitfun(vs, x):
                return vs[0]*x

            def errfun(vs, x, y):
                return fitfun(vs, x) - y

            sub_proto = self.protovector(
                remove_group, trange=trange, rectify=rectify)
            [vscalc, success] = optimize.leastsq(
                errfun, [0.5], args=(sub_proto, unit))
            unit -= vscalc[0]*sub_proto
            unit /= nansum(np.abs(unit))

        return unit

    def meanresp(self, trange=(0, 2), rectify=False, hz=None):
        """
        Get the mean responses to each of the groups.

        :param trange: time range in seconds
        :param rectify: set values < 0 equal to 0 if True
        :param hz: the frequency of the recording, will automatically find if necessary
        :return: a dict of the unit vectors for each group
        """
        if hz is not None:
            # 190207
            warnings.warn(
                "Pass framerate argument to the GLM init.",
                DeprecationWarning)
        out = {}
        for group in self.groups():
            vec = self.basis(group, trange, hz=hz)  # ncells x ntimes
            if rectify:
                vec[vec < 0] = 0
            out[group] = vec

        return out

    def labels(self, minpred=0.01, minfrac=0.05):
        """
        Label cells by their GLM filter responses.

        :param minpred: the minimum variance predicted by all glm filters
        :param minfrac: the minimum fraction of glm variance explained by filters for each cell type
        :return: dict of labels
        """

        # Get cell groups and the sufficiency of each deviance explained
        groupdev = (self.devexp[:, 1:].T*(self.devexp[:, 0] >= minpred)).T
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

    def explained(self, scale_by_total=True):
        """
        Get the deviance explained by each group of behavioral vectors.

        Parameters
        ----------
        scale_by_total : bool
            If true, scale the deviance explained by the total deviance explained
            for this cell. Makes the values comparable between cells.

        Returns
        -------
        dict of numpy vectors of ncells
            Dictionary of deviance explained by cell group as a fraction of total deviance explained
            Also includes a value of total deviance explained

        """

        if scale_by_total:
            groupdev = self.devexp[:, 1:]
        else:
            groupdev = (self.devexp[:, 1:].T*(self.devexp[:, 0])).T

        odict = {'total': self.devexp[:, 0]}
        for i, name in enumerate(self.cellgroups):
            odict[name] = groupdev[:, i]

        return odict
