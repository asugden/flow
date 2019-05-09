"""Randomize in time."""
import numpy as np

from ..base_classifier import BaseClassifier
from ... import paths
from ...misc import loadmat


class RandomizeTime(BaseClassifier):
    """Time randomizer."""

    def __init__(
            self, parent, nrandomizations=10, mask_running=True,
            mask_licking=True, mask_motion=True):
        """Time randomizer init.

        Parameters
        ----------
        nrandomizations : int
            Number of randomizations to perform.
        mask_running : bool
            If True, mask out running times.
        mask_licking : bool
            If True, mask out licking times.
        mask_motion : bool
            If True, mask out times of high brain motion.

        """
        BaseClassifier.__init__(self)

        self.nrand = nrandomizations
        self.mask_running = mask_running
        self.mask_licking = mask_licking
        self.mask_motion = mask_motion

        self.d = None
        self._rand_traces = None
        self._inactivity = None
        self._no_inactivity_found = False

        self.parent = parent
        self.pars = parent.pars

        randomization_str = 'time'
        if not self.mask_running:
            randomization_str += '_no_run_mask'
        if not self.mask_licking:
            randomization_str += '_no_lick_mask'
        if not self.mask_motion:
            randomization_str += '_no_mot_mask'

        path = paths.getc2p(parent.run.mouse, parent.run.date, parent.run.run,
                            self.pars, randomization_str, self.nrand)

        try:
            self.d = loadmat(path)
        except IOError:
            self._classify(path)

    def real_false_positives(
            self, cs, threshold=0.1, xmask=True, max=2, downfor=2, maxlen=-1,
            fmin=-1, saferange=(-5, 5)):
        """
        Return the number of real and false positives for a given stimulus.

        Parameters
        ----------
        cs : str
            Stimulus name, e.g. plus
        threshold : float
            The classifier threshold used to identify events
        xmask : boolean
            If True, events cannot be found in more than one non-'other'
            category.
        max : float
            Maximum classifier value to accept
        downfor : int
            Number of frames in which a reactivation event cannot be identified
        maxlen : int
            Only return those events that are above threshold for less long
            than maxlen.
        fmin : int
            Minimum frame number allowed (for masking beginning of recordings)
        saferange : tuple of ints
            A frame range over which events can be identified

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

        mask = self.inactivity()

        real = self.parent.events(
            cs, threshold, trs, mask=mask, xmask=xmask, max=max,
            downfor=downfor, maxlen=maxlen, fmin=fmin, saferange=saferange)

        rand = self.events(
            cs, threshold, self._traces(self.d['shifts']), mask=None,
            xmask=xmask, max=max, downfor=downfor, maxlen=maxlen, fmin=fmin,
            saferange=saferange)

        return len(real), len(rand)/float(self.nrand)

    def _classify(self, path):
        """Run a randomized analysis."""

        t2p = self.parent.run.trace2p()

        if np.sum(self.inactivity()) <= 201:
            self._no_inactivity_found = True
            return

        shifts = np.zeros((t2p.ncells, self.nrand), dtype=bool)
        for r in range(self.nrand):
            for c in range(t2p.ncells):
                # randint is left inclusive and right exclusive
                shifts[c, r] = np.random.randint(0, t2p.nframes)
        out = self._traces(shifts)
        self.d = self.parent.classify(data=out)
        self.d['shifts'] = shifts

        self._save(path)

    def _traces(self, shifts):
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
                mask = self.inactivity()
                trs = trs[:, mask]

                for c in range(t2p.ncells):
                    trs[c, :] = np.roll(trs[c, :], shifts[c, r])

                if len(self._rand_traces) == 0:
                    self._rand_traces = trs
                else:
                    self._rand_traces = np.concatenate(
                        [self._rand_traces, trs], 1)

        return self._rand_traces

    def inactivity(self):
        """Return inactivity mask."""
        if self._inactivity is None:
            kwargs = {}

            # Mask each stimuli on training runs, and up to last stim for
            # any other run type
            if self.parent.run.run_type == 'training':
                kwargs['nostim'] = 'each'
                kwargs['pre_pad_s'] = self.pars['classification-ms']/1000./2.
                kwargs['post_pad_s'] = 0.0 + kwargs['pre_pad_s']
                kwargs['pav_post_pad_s'] = 0.5 + kwargs['pre_pad_s']
            else:
                kwargs['nostim'] = 'last'

            if not self.mask_running:
                kwargs['runsec'] = -1
            if not self.mask_licking:
                kwargs['licksec'] = -1
            if not self.mask_motion:
                kwargs['motsec'] = -1

            mask = self.parent.run.trace2p().inactivity(**kwargs)

            self._inactivity = mask
        return self._inactivity
