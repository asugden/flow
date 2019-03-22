from builtins import range
from builtins import object
from copy import deepcopy
import numpy as np

from .misc import loadmat
from classifier import train


class Classify2P(object):
    def __init__(self, path, run, pars):
        """
        Load in a classifier or classifiers

        Parameters
        ----------
        paths : str or list
            A single path or a list of paths to load
        pars : dict
            The parameters used to generate the classifier
        """

        self.pars = pars
        self.run = run
        self._xresults = None
        self._trained_model = None
        self._trained_params = None
        self._trained_activity = None

        try:
            self.d = loadmat(path)
        except IOError:
            self._classify(path)

    def __repr__(self):
        return "Classify2P(paths={})".format(self._paths)

    def classify(self, data, priors=None, temporal_prior=None, integrate_frames=1):
        """
        Return a trained classifier either for running the traditional classifier
        or for randomization.

        data : matrix
            Matrix of data to compare, ncells x ntimes
        priors : dict
            The prior probabilities for each class. Defaults to pars.
        temporal_prior : vector
            A vector of weights per unit time. Defaults to the standard
            temporal prior if set in pars.
        integrate_frames : int
            The number of frames to integrate.

        Returns
        -------
        dict
            The standard output format of train.py

        """

        # Train only once
        if self._trained is None:
            self._trained_model, self._trained_params, self.trained_activity = \
                train.train_classifier(self.run, **self.pars)

        results = train.classify_reactivations(
            run, self._trained_model, self._trained_params, self._trained_activity)

        return results

    def results(self, cs='', xmask=True):
        """
        Return the classifier results.

        Parameters
        ----------
        cs : str
            Optional, return a vector if string, otherwise return result dict
        xmask : bool
            If true, force it so that there can be only one event in any non-other category

        Returns
        -------
        ndarray
            Of size ntimes if cs is a string, otherwise nclasses x ntimes

        """

        if len(cs) and cs not in self.d['results']:
            return []

        if xmask:
            self._xmask()

            if cs == '':
                return self._xresults
            else:
                return self._xresults[cs]
        else:
            if cs == '':
                return self.d['results']
            else:
                return self.d['results'][cs]

    def events(self, cs, threshold, traces, mask=None, xmask=True, max=2,
               downfor=2, maxlen=-1, fmin=-1, saferange=(-5, 5)):
        """
        Find all reactivation events for stimulus type.

        Parameters
        ----------
        cs : str
            Stimulus name, e.g. plus
        traces : ndarray of ncells x ntimes or Trace2P instance
            Activity traces from which the peak population activity is derived
        threshold : float
            The classifier threshold used to identify events
        mask : boolean ndarray
            A mask of which times to search for events over
        xmask : boolean
            If true, events cannot be found in more than one non-'other' category
        max : float
            Maximum classifier value to accept
        downfor : int
            Number of frames in which a reactivation event cannot be identified
        maxlen : int
            Only return those events that are above threshold for less long than maxlen
        fmin : int
            Minimum frame number allowed (for masking beginning of recordings)
        saferange : tuple of ints
            A frame range over which events can be identified

        Returns
        -------
        ndarray
            A list of frame numbers on which the peak population activity was detected

        """
        if cs not in self.d['results']:
            return []

        if xmask:
            self._xmask()

        res = np.copy(self.d['results'][cs]) if not xmask else np.copy(self._xresults[cs])
        if mask is not None:
            res[np.invert(mask)] = 0

        return peaks(res, traces, threshold, max, downfor, maxlen, fmin, saferange)

    def randomization(self, rtype):
        """
        Return an object of the correct randomization type.

        Parameters
        ----------
        rtype : str {'identity', 'circshift'}
            Randomization type

        Returns
        -------
        object
            Randomization object

        """

        pass

    def _xmask(self):
        """Mask results such that events can only be found in one non-other cs."""

        if self._xresults is None:
            nonother = [key for key in self.d['results'] if 'other' not in key]
            maxmat = np.zeros((len(nonother), len(self.d['results'][nonother[0]])))
            for i, cs in enumerate(nonother):
                maxmat[i, :] = self.d['results'][cs]
            maxact = np.nanmax(maxmat, axis=0)

            self._xresults = deepcopy(self.d['results'])
            for cs in nonother:
                self._xresults[cs][self._xresults[cs] < maxact] = 0

    def _classify(self):
        raise NotImplementedError('Need to add classification.')


def count(result, threshold, all=False, max=2, downfor=2, offsets=False):
    """Count the number of replay events of non-"other" stimuli.

    Replay events are defined has having a probability greater than
    threshold. Threshold should not be > 0.5. If skip_sequential is
    false, a replay event lasting multiple frames will be counted as
    multiple events.

    :param result: classifier vector for a single cs, e.g. classifier['results']['plus']
    :param threshold: the minimum value above which it will be classified as a replay event
    :param all: include all times above threshold if True, otherwise once per crossing
    :param max: the maximum value to allow
    :param downfor: the number of frames during which the classifier has to dip below threshold
    :param offsets: return the onsets as well as offsets
    :return: a vector of frames in which the value went above threshold

    """
    # Flatten values less than threshold (set to 0)
    flat = np.copy(result)
    flat[flat <= threshold] = 0
    flat[flat > 0] = 1

    # In one case, bail immediately
    if max >= 1.0 and all: return np.arange(len(flat))[flat > 0]

    # Smooth over onsets
    for i in range(downfor, 0, -1):
        skipns = np.convolve(flat, [1] + [0]*i + [1], 'same')
        flat[skipns == 2] = 1

    # Event onsets
    ons = np.concatenate([[flat[0]], np.diff(flat)])
    offs = np.concatenate([np.diff(flat), [-1*flat[-1]]])
    ons = np.arange(len(ons))[ons > 0]
    offs = np.arange(len(offs))[
               offs < 0] + 1  # Before the + 1, the off was the last frame, not the following frame

    # Now we can account for the case in which there is no max set
    if max >= 1.0:
        if offsets:
            return ons, offs
        else:
            return ons

    # Account for the maximum
    out = []
    outoffs = []
    for i in range(len(ons)):
        if np.max(result[ons[i]:offs[i]]) < max:
            if all:
                out.extend(list(range(ons[i], offs[i])))
            else:
                out.append(ons[i])
                outoffs.append(offs[i])

    if not all and offsets:
        return np.array(out), np.array(outoffs)
    else:
        return np.array(out)


def peaks(result, trs, threshold, max=2, downfor=2, maxlen=-1, fmin=-1, saferange=(-5, 5)):
    """Return the times of peak activity of replay events found by counts

    :param result: classifier vector for a single cs, e.g. classifier['results']['plus']
    :param trs: a Trace2P instance from which to determine population activity or a matrix of ncells, nframes
    :param threshold: the minimum value above which it will be classified as a replay event
    :param max: the maximum value to allow
    :param downfor: the number of frames during which the classifier has to dip below threshold
    :param maxlen: only return those events that are above threshold for less long than maxlen
    :param fmin: minimum frame allowed
    :param saferange: a tuple of values which need to be safe around which replays can be integrated
    :return: a vector of events in which the value went above threshold centered on max population activity

    """
    # Check if trs is a trace2p instance or whether it is a matrix
    t2p = trs
    if not hasattr(trs, "__len__"): trs = t2p.trace('deconvolved')

    # Get all onsets and offsets
    out = []
    evs, evoffs = count(result, threshold, False, max, downfor, True)

    # Find the point in each event with the highest deconvolved activity
    for i in range(len(evs)):
        if maxlen < 1 or evoffs[i] - evs[i] < maxlen:
            act = np.nanmean(trs[:, evs[i]:evoffs[i]], axis=0)
            peakpos = evs[i] + np.argmax(act)

            if (peakpos > fmin
                    and abs(saferange[0]) < peakpos < np.shape(trs)[1] - abs(saferange[1])):
                out.append(peakpos)

    return out


def peakprobs(result, threshold, max=2, downfor=2, maxlen=-1):
    """
    Return the times of peak classification probability of replay events found by counts

    :param result: classifier vector for a single cs, e.g. classifier['results']['plus']
    :param threshold: the minimum value above which it will be classified as a replay event
    :param max: the maximum value to allow
    :param downfor: the number of frames during which the classifier has to dip below threshold
    :param maxlen: only return those events that are above threshold for less long than maxlen
    :return: a vector of events in which the value went above threshold centered on max population activity
    """

    # Get all onsets and offsets
    out = []
    evs, evoffs = count(result, threshold, False, max, downfor, True)

    # Find the point in each event with the highest deconvolved activity
    for i in range(len(evs)):
        if maxlen < 1 or evoffs[i] - evs[i] < maxlen:
            act = result[evs[i]:evoffs[i]]
            out.append(evs[i] + np.argmax(act))

    return out
