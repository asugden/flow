import math
import numpy as np

from .. import train
from ..aode import rollingmax
from ..base_classifier import BaseClassifier
from ... import paths
from ...misc import loadmat


class RandomizeIdentity(BaseClassifier):
    def __init__(self, parent, nrandomizations=100, inactivity_mask=True):
        BaseClassifier.__init__(self)

        self.parent = parent
        self.pars = parent.pars

        randomization_str = 'identity'
        if not inactivity_mask:
            randomization_str += '_no_inact_mask'

        path = paths.getc2p(parent.run.mouse, parent.run.date, parent.run.run,
                            self.pars, randomization_str, nrandomizations)

        self.d = None
        self._rand_traces = None
        self._rand_prior = None
        self._nrand = float(nrandomizations)
        self._no_events_found = False

        try:
            self.d = loadmat(path)
            if isinstance(self.d['event_thresholds'], float):
                self.d['event_thresholds'] = [self.d['event_thresholds']]
                self.d['event_stimuli'] = [self.d['event_stimuli']]
                self.d['event_frames'] = [self.d['event_frames']]
        except IOError:
            self._classify(path, nrandomizations, inactivity_mask)

    def real_false_positives(
            self, cs, threshold=0.1, matching_cs=False, max=2, downfor=2,
            maxlen=-1, fmin=-1, saferange=(-5, 5), inactivity_mask=True):
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
        if self._no_events_found:
            return 0, 0

        matching = np.array([True if str(stim).strip() == cs and
                             threshold <= thresh < max else False
                             for stim, thresh in zip(self.d['event_stimuli'],
                                                     self.d['event_thresholds'])])

        real = int(np.sum(matching))
        matching = np.concatenate([matching for _ in range(int(self._nrand))])

        if real == 0:
            return 0, 0.0

        best_match = self.d['results'][cs][matching]
        if not matching_cs:
            for rcs in self.d['results'].keys():
                if 'other' not in rcs and rcs != cs:
                    best_match = np.max([best_match, self.d['results'][rcs][matching]], axis=0)

        rand = np.sum(best_match >= threshold)/self._nrand

        return real, rand

    def _classify(self, path, nrand, inactivity_mask):
        """Run a randomized analysis."""

        t2p = self.parent.run.trace2p()
        cses = [key for key in self.parent.d['results'] if 'other' not in key]

        if inactivity_mask:
            mask = t2p.inactivity()
        else:
            mask = None

        evs, thresholds, stimuli = [], [], []
        for cs in cses:
            for t in np.arange(0.05, 1.01, 0.05)[::-1]:
                tevents = self.parent.events(cs, t, mask=mask)
                tevents = np.setdiff1d(tevents, evs).astype(np.int32)
                tthresh = np.ones(len(tevents))*t
                evs = np.concatenate([evs, tevents])
                thresholds = np.concatenate([thresholds, tthresh])
                stimuli.extend([cs]*len(tevents))

        if len(evs) == 0:
            self._no_events_found = True
            return

        order = np.argsort(evs)
        evs, thresholds = evs[order].astype(np.int32), thresholds[order]
        stimuli = [stimuli[v] for v in order]

        orders = np.zeros((t2p.ncells, nrand), dtype=np.int32)

        cell_array = np.arange(t2p.ncells, dtype=np.int32)
        for r in range(nrand):
            np.random.shuffle(cell_array)
            orders[:, r] = cell_array

        tpriorvec = train.temporal_prior(
            self.parent.run, self.pars, nan_cells=None)

        trs, temporal = self._traces(evs, orders, tpriorvec)

        self.d = self.parent.classify(
            data=trs, temporal_prior=temporal, integrate_frames=1)
        self.d['event_frames'] = evs
        self.d['event_thresholds'] = thresholds
        self.d['event_stimuli'] = stimuli
        self.d['orders'] = orders

        self._save(path)

    def _traces(self, evs, orders, tprior):
        """
        Return or reconstruct all of the traces used for randomization.

        Returns
        -------
        Traces, ncells x ntimes
        """

        if self._rand_traces is None:
            t2p = self.parent.run.trace2p()
            trs = np.copy(t2p.trace('deconvolved'))
            integrate_frames = int(round(self.pars['classification-ms']/
                                         1000.0*t2p.framerate))
            trs = rollingmax(trs, integrate_frames)
            rollframes = t2p.nframes - (integrate_frames - 1)
            frame_range = (int(math.floor(integrate_frames/2.0)),
                           rollframes + int(math.floor(integrate_frames/2.0)))

            self._rand_traces, self._rand_prior = [], []
            for r in range(np.shape(orders)[1]):
                for ev in evs:
                    ev -= frame_range[0]  # Offset for rolling frames

                    self._rand_traces.append(trs[orders[:, r], ev])
                    self._rand_prior.append(tprior[ev])

            self._rand_traces = np.array(self._rand_traces).T
            self._rand_prior = np.array(self._rand_prior)

        return self._rand_traces, self._rand_prior
