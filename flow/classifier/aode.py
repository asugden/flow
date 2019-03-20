from __future__ import division, print_function
from builtins import object, range

import math
import numpy as np

import runclassifier


class AODE(object):
    """
    Create a generative model based on binarized spiking data. Use an
    average one-dependency estimator (AODE) to calculate the P(cs|data).
    Relies on estimating the probability of replay.
    """

    def __init__(self, cses, priors, nframes):
        """Pass in deconvolved data for training.
        It is essential that cses are >= 0 and <= 1!
        cses are nonsets, ncells, nframes.
        """

        self._cond = {}  # Used to be _gmpjoint, now conditional
        self._marg = {}  # Used to be _gmpsingle, now marginal
        self._pseudocount = 0.1
        self._classnames = []

        self.ncells = None

    def train(self, tevents, classifier='aode'):
        """
        Train a classifier by measuring the probabilities of single-cell
        and pairwise firing over a range of frames relative to the stimuli.

        Parameters
        ----------
        tevents : matrix
            Deconvolved values, ranged from 0 to 1 of the shape
            nonsets x ncells x nframes
        classifier : str {'aode', 'naivebayes'}
            Change the classifier from AODE to naive bayes
        """

        # Save the classnames in an order because dicts can change.
        self._classnames = [key for key in tevents]
        self.ncells = np.shape(tevents[self._classnames[0]])[1]

        if classifier == 'naivebayes':
            return self._naivebayes(tevents)

        for condition in self._classnames:
            # Take the max across frames
            stims = np.max(tevents[condition], axis=2).T
            stiminv = 1.0 - stims
            # Stims is now in shape nonsets, ncells  NOT ncells, nonsets
            nonsets = np.shape(stims)[0]  # NOTE: WAS 1 UNTIL 190211

            # List of probabilities of doublet and singlet spiking of
            # size (matching cells, total cells, 6)
            self._cond[condition] = np.zeros((self.ncells, self.ncells, 4))
            self._marg[condition] = np.zeros((self.ncells, 2))

            # Calculate single probabilities
            self._marg[condition][:, 0] = np.sum(stims, axis=1)
            self._marg[condition][:, 1] = np.sum(stiminv, axis=1)

            # For every matching cell, calculate the probabilities
            for c in range(self._ncells):
                # Repeat the value for each cell to make a tiled array of cell c
                crep = np.tile(stims[c, :], self.ncells).reshape((self.ncells, nonsets))
                crepinv = np.tile(stiminv[c, :], self.ncells).reshape((self.ncells, nonsets))

                self._cond[condition][c, :, 0] = np.sum(crep*stims, 1)  # TT
                self._cond[condition][c, :, 1] = np.sum(crep*stiminv, 1)  # TF

                self._cond[condition][c, :, 2] = np.sum(crepinv*stims, 1)  # FT
                self._cond[condition][c, :, 3] = np.sum(crepinv*stiminv, 1)  # FF

                # Set the joint of the same cell equal to 0 so that it's not included
                self._cond[condition][c, c, :] = 0

            # Add pseudocounts
            self._cond[condition] += self._pseudocount
            self._marg[condition] += self._pseudocount*2

            # Divide by the number of onsets
            self._cond[condition] /= float(np.shape(stims)[1] + 4*self._pseudocount)
            self._marg[condition] /= float(np.shape(stims)[1] + 4*self._pseudocount)

            # Divide by the marginal
            for c in range(self.ncells):
                self._cond[condition][c, :, 0] /= self._marg[condition][c, 0]
                self._cond[condition][c, :, 1] /= self._marg[condition][c, 0]
                self._cond[condition][c, :, 2] /= self._marg[condition][c, 1]
                self._cond[condition][c, :, 3] /= self._marg[condition][c, 1]

            # 0, P(x_i == T | x_j == T) = P(TT)/P(x_j == T)
            # 1, P(x_i == T | x_j == F)
            # 2, P(x_i == F | x_j == T)
            # 4, P(x_j == T)
            # 5, P(x_j == F)

        return self

    def compare(self, data, integrate_frames, priors, naive_bayes=False):  #, priors={}, widenpriors=True, outliers=None):
        """
        Run the comparison using a numpy extension written in C.
        speed.

        Parameters
        ----------
        data : matrix
            Data of type ncells x ntimes
        integrate_frames : int
            The number of frames to take the max over, usually 4
        priors : dict of vectors
            The prior probability of each class type. Note:
            PRIORS MUST SUM TO 1! Assumes that one is using assign_temporal_priors
        naive_bayes : bool
            If True, run comparison as Naive Bayes rather than AODE.

        Returns
        -------
        dict of vectors
            The probability of reactivation in each case.

        """

        # Double-check that the data has the correct number of cells
        ncells, nframes = np.shape(data)
        if ncells != self.ncells:
            raise ValueError('Wrong number of cells in dataset')

        # Double-check that priors have been set
        if not set(self._classnames).issubset(set(priors.keys())):
            raise ValueError('Not all classes have priors set')

        if self._class_probabilities == {}:
            raise IndexError('Have not yet set priors')

        # Set the correct sizes based on the frame integration
        if integrate_frames > 1:
            rollframes = nframes - (integrate_frames - 1)
            frame_range = (int(math.floor(integrate_frames/2.0)),
                rollframes + int(math.floor(integrate_frames/2.0)))
            data = rollingmax(data, integrate_frames)
        else:
            rollframes = nframes
            frame_range = (0, nframes)

        if not self._nb:
            # Convert dicts to arrays for the numpy extension
            sprobs, jprobs, likelihood, res = self._prob_dict_to_np(rollframes)
            cprobs = np.array([priors[key][frame_range[0]:frame_range[1]] for key in self._classnames])

            # Run AODE
            runclassifier.aode(sprobs, jprobs, cprobs, data, res, likelihood)
        else:
            sprobs, likelihood, res = self._prob_dict_to_np(rollframes, True)
            cprobs = np.array([priors[key][frame_range[0]:frame_range[1]] for key in clses])

            # Run Naive Bayes
            runclassifier.naivebayes(sprobs, cprobs, data, res, likelihood)

        # Copy output into the appropriate style
        out = {}
        likely = {}
        for i, key in enumerate(self._classnames):
            out[key] = np.zeros(nframes)
            out[key][frame_range[0]:frame_range[1]] = res[i]
            likely[key] = likelihood[i]

        return out, data, likely

    def describe(self):
        """
        List the key bits of information about each stimulus as a string
        that is returned.
        """

        out = '%13s    %5s    %6s   %5s    %5s    %5s    %5s    %5s    %5s    %5s  %5s  %6s\n'%(
            'Stimulus', 'Mean', 'Median', 'SMax', 'JMax', 'SSum', 'JSumTT', 'JSumTF', 'JSumFT', 'JSumFF', 'Cells',
            'Onsets')
        for cs in self.d:
            out += '%13s    %.3f    %.3f    %.3f    %.3f    %5.2f    %6.0f    %6.0f    %6.0f    %6.0f    %3i    %4i\n'%(
                cs,
                np.mean(self._marg[cs][:, 0]),
                np.median(self._marg[cs][:, 0]),
                np.max(self._marg[cs][:, 0]),
                np.max(self._cond[cs][:, :, 0]) if not self._nbonly else 0,
                np.sum(self._marg[cs][:, 0]),
                np.sum(self._cond[cs][:, :, 0]) if not self._nbonly else 0,
                np.sum(self._cond[cs][:, :, 2]) if not self._nbonly else 0,
                np.sum(self._cond[cs][:, :, 1]) if not self._nbonly else 0,
                np.sum(self._cond[cs][:, :, 3]) if not self._nbonly else 0,
                len(self._marg[cs][:, 0]),
                np.shape(self.d[cs])[0],
            )
        return out

    def _prob_dict_to_np(self, nframes, naive_bayes=False):
        """
        Convert the dict of probabilities into a single numpy array to
        pass to the C numpy extension.
        """

        # Get a list of classes for results
        clses = self._classnames
        k = clses[0]

        # Allocate and fill each array
        sprobs = np.zeros((len(clses), np.shape(self._marg[k])[0], 2), dtype=np.float64)
        if not naive_bayes:
            jprobs = np.zeros((len(clses), np.shape(self._cond[k])[0],
                               np.shape(self._cond[k])[1], 4), dtype=np.float64)
        likelihood = np.zeros((len(clses), nframes), dtype=np.float64)
        res = np.zeros((len(clses), nframes), dtype=np.float64)

        for i, key in enumerate(clses_wo_baseline):
            sprobs[i] = self._marg[key]
            if not naive_bayes:
                jprobs[i] = self._cond[key]

        if not naive_bayes:
            return sprobs, jprobs, likelihood, res
        else:
            return sprobs, likelihood, res


def rollingmax(arr, integrate_frames):
    """
    Get the rolling maximum across the final axis for 1d or 2d array arr, which will be converted to double
    :param arr: 1d or 2d array of doubles
    :param integrate_frames: number of frames to integrate across the last axis, int
    :return: arr with the final axis of length input - (integrate_frames - 1)
    """

    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)

    if arr.ndim == 1:
        out = np.zeros(len(arr) - (integrate_frames - 1))
    elif arr.ndim == 2:
        out = np.zeros((np.shape(arr)[0], np.shape(arr)[1] - (integrate_frames - 1)))
    else:
        raise ValueError('Function only handles 1d and 2d arrays')
    print((arr.dtype, out.dtype))
    runclassifier.rollmax(arr, out, integrate_frames)
    return out

def rollingmean(arr, integrate_frames):
    """
    Get the rolling maximum across the final axis for 1d or 2d array arr, which will be converted to double
    :param arr: 1d or 2d array of doubles
    :param integrate_frames: number of frames to integrate across the last axis, int
    :return: arr with the final axis of length input - (integrate_frames - 1)
    """

    if arr.dtype != np.float64:
        arr = arr.astype(np.float64)

    if arr.ndim == 1:
        out = np.zeros(len(arr) - (integrate_frames - 1))
    elif arr.ndim == 2:
        out = np.zeros((np.shape(arr)[0], np.shape(arr)[1] - (integrate_frames - 1)))
    else:
        raise ValueError('Function only handles 1d and 2d arrays')

    runclassifier.rollmean(arr, out, integrate_frames)
    return out

def temporal_prior(traces, actmn, actvar, fwhm, expand=1):
    """
    Generate temporal-dependent priors using basis sets and mexican-hat functions
    :param traces: matrix of traces, ncells by nframes
    :param actmn: mean activity
    :param actvar: variation above which we will consider it a guaranteed event
    :param fwhm: the full-width at half-maximum to use for the temporal prior
    :param expand: use a moving max across expand frames
    :return: prior vector
    """

    from scipy.stats import norm

    # Set the half-width of the convolution kernel
    xhalfwidth = 100

    # Determine a normal function sigma from the full-width at half-maximum
    def sigma(fwhm_):
        return fwhm_/(2*np.sqrt(2*np.log(2)))

    # Generate the basis functions and correct population activity for baseline and variation
    basis = np.power(fwhm, np.arange(4) + 1)
    popact = (np.nanmean(traces, axis=0) - actmn)/actvar
    fits = np.zeros((len(basis) - 1, len(popact)))

    # Get the first basis normal function
    defrange = int(norm.interval(0.99999, loc=0, scale=sigma(basis[0]))[1]) + 3
    defrange = min(xhalfwidth, defrange)
    b0 = np.zeros(2*xhalfwidth + 1)
    b0[xhalfwidth - defrange:xhalfwidth + defrange + 1] = norm.pdf(
        range(-defrange, defrange + 1), loc=0, scale=sigma(basis[0]))

    # Generate the fits
    for b in range(1, len(basis)):
        defrange = int(norm.interval(0.99999, loc=0, scale=sigma(basis[b]))[1]) + 3
        defrange = min(xhalfwidth, defrange)
        bn = np.zeros(2*xhalfwidth + 1)
        bn[xhalfwidth - defrange:xhalfwidth + defrange + 1] = norm.pdf(
            range(-defrange, defrange + 1), loc=0, scale=sigma(basis[b]))
        fits[b-1, :] = np.convolve(popact, b0 - bn, 'same')

    # And return the wfits to the narrowest basis function
    weights = np.clip(np.nanmin(fits, axis=0), 0, 1)

    # Use a running max if expand > 1
    if expand > 1:
        weights = rollingmax(weights, expand).flatten()

    return weights

def assign_temporal_priors(priors, tprior, keyword='other'):
    """
    Apply the temporal prior to a prior dictionary, applying the temporal prior to any member of the dict without
    keyword
    :param priors: temporally-independent priors which will be combined with, anything with 'other' in it will get
    the inverse of the temporal prior, while the remaining groups will get multiplied with the temporal prior
    :param tprior: the temporal prior, ouput from temporal_prior()
    :param keyword: keyword for which dict values to exclude from temporal prior
    :return: dict of priors
    """

    # Normalize priors
    psum = 0
    for key in priors: psum += priors[key]

    # Iterate over priors, accounting for "others" and non-"others" differently
    rclses = [key for key in priors if keyword not in key]
    oclses = [key for key in priors if keyword in key]
    rpriors = np.zeros((len(rclses), len(tprior)))
    for i, key in enumerate(rclses): rpriors[i, :] = priors[key]/psum*tprior

    # Get the remaining probability and add to others
    opriorsum = np.sum([priors[key]/psum for key in oclses])
    divvy = 1.0 - opriorsum - np.sum(rpriors, axis=0)
    opriors = np.zeros((len(oclses), len(tprior)))
    for i, key in enumerate(oclses): opriors[i, :] = (priors[key]/psum)*(1.0 + 1.0/opriorsum*divvy)

    # Put into dict and return
    out = {}
    for i, key in enumerate(rclses): out[key] = rpriors[i, :]
    for i, key in enumerate(oclses): out[key] = opriors[i, :]
    return out

def classify(data, priors, integrate_frames):
    """
    Return a classifier class ready to be trained.
    :param data: Deconvolved or binarized calcium data
    :param priors: dict of stimulus types and prior probabilities
    :param integrate_frames: number of frames to integrate for comparison data
    :return: classifier class
    """

    out = AODE(data, priors, integrate_frames)
    return out


if __name__ == '__main__':
    print(rollingmean(np.arange(20), 3))