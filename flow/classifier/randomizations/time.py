"""Randomize in time."""
import numpy as np

from ..base_classifier import BaseClassifier
from ... import paths
from ...misc import loadmat


class RandomizeTime(BaseClassifier):
    def __init__(self, parent, nrandomizations=10, inactivity_mask=True):
        BaseClassifier.__init__(self)

        self.parent = parent
        self.pars = parent.pars

        randomization_str = 'time'
        if not inactivity_mask:
            randomization_str += '_no_inact_mask'

        path = paths.getc2p(parent.run.mouse, parent.run.date, parent.run.run,
                            self.pars, randomization_str, nrandomizations)

        self.d = None
        self._rand_traces = None
        self._nrand = float(nrandomizations)
        self._no_inactivity_found = False

        try:
            self.d = loadmat(path)
        except IOError:
            self._classify(path, nrandomizations, inactivity_mask)

    def real_false_positives(
            self, cs, threshold=0.1, xmask=True, max=2, downfor=2, maxlen=-1,
            fmin=-1, saferange=(-5, 5), inactivity_mask=True):
        """
        Return the number of real and false positives for a given stimulus.

        Parameters
        ----------
        cs : str
            Stimulus name, e.g. plus
        threshold : float
            The classifier threshold used to identify events
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
        inactivity_mask : bool
            If True, mask out active times.

        Returns
        -------
        real : int
            Number of real events detected.
        rand : float
            Mean number of random events per randomization.

        """
        if self._no_inactivity_found:
            return np.nan, np.nan

        t2p = self.parent.run.trace2p()
        trs = t2p.trace('deconvolved')
        if inactivity_mask:
            mask = t2p.inactivity()
        else:
            mask = None

        real = self.parent.events(cs, threshold, trs, mask=mask, xmask=xmask,
                                  max=max, downfor=downfor, maxlen=maxlen,
                                  fmin=fmin, saferange=saferange)

        rand = self.events(cs, threshold, self._traces(self.d['shifts']),
                           mask=None, xmask=xmask, max=max, downfor=downfor,
                           maxlen=maxlen, fmin=fmin, saferange=saferange)

        return len(real), len(rand)/self._nrand

    def _classify(self, path, nrand, inactivity_mask):
        """Run a randomized analysis."""

        t2p = self.parent.run.trace2p()

        if inactivity_mask and np.sum(t2p.inactivity()) <= 201:
            self._no_inactivity_found = True
            return

        shifts = np.zeros((t2p.ncells, nrand), dtype=bool)
        for r in range(nrand):
            for c in range(t2p.ncells):
                # randint is inclusive for both beginning and end
                shifts[c, r] = np.random.randint(0, t2p.nframes - 1)
        out = self._traces(shifts, inactivity_mask)
        self.d = self.parent.classify(data=out)
        self.d['shifts'] = shifts

        self._save(path)

    def _traces(self, shifts, inactivity_mask):
        """
        Return or reconstruct all of the traces used for randomization.

        Returns
        -------
        Traces, ncells x ntimes
        """

        if self._rand_traces is None:
            t2p = self.parent.run.trace2p()

            self._rand_traces = []
            for r in range(np.shape(shifts)[1]):
                trs = np.copy(t2p.trace('deconvolved'))
                if inactivity_mask:
                    mask = t2p.inactivity()
                    trs = trs[:, mask]

                for c in range(t2p.ncells):
                    trs[c, :] = np.roll(trs[c, :], shifts[c, r])

                if len(self._rand_traces) == 0:
                    self._rand_traces = trs
                else:
                    self._rand_traces = np.concatenate(
                        [self._rand_traces, trs], 1)

        return self._rand_traces
