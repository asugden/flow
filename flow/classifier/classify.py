# Intended to be temporary and eventually replaced with
# classsify_reactivations and train_classifier: 180821
from copy import deepcopy
import datetime
import math
import numpy as np
import os.path as opath
from scipy.io import savemat
from scipy.stats import norm

# This dependency should really be removed
# Moved into function for now
# from pool import database

from .. import outfns, paths
from .. import metadata as metadata
from . import aode, randomizations
from ..misc import legiblepars


def temporal_prior(traces, actmn, actvar, outliers, fwhm, thresh, priors, expand=1):
    """
    Generate temporal-dependent priors using basis sets and mexican-hat functions

    :param traces: matrix of traces, ncells by nframes
    :param actmn: mean activity
    :param actvar: variation above which we will consider it a guaranteed event
    :param thresh: threshold at which to binarize temporal prior
    :param priors: temporally-independent priors which will be combined with
    :param expand: use a moving max across expand frames
    :return: prior vectors
    """

    # Set the half-width of the convolution kernel
    xhalfwidth = 100

    # Determine a normal function sigma from the full-width at half-maximum
    def sigma(fwhm_): return fwhm_/(2*np.sqrt(2*np.log(2)))

    # Generate the basis functions and correct population activity for baseline and variation
    basis = np.power(fwhm, np.arange(4) + 1)
    popact = (np.nanmean(traces[np.invert(outliers), :], axis=0) - actmn)/actvar
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
        bn[xhalfwidth - defrange:xhalfwidth + defrange + 1] = norm.pdf(range(-defrange, defrange + 1), loc=0, scale=sigma(basis[b]))
        fits[b-1, :] = np.convolve(popact, b0 - bn, 'same')

    # And return the wfits to the narrowest basis function
    weights = np.clip(np.nanmin(fits, axis=0), 0, 1)

    # Use a running max if expand > 1
    if expand > 1:
        weights = outfns.movingmax(weights, expand).flatten()

    # Normalize priors
    psum = float(np.sum([priors[key] for key in priors]))
    priors = {key: priors[key]/psum for key in priors}
    psum = 1

    # Iterate over priors, accounting for "others" and non-"others" differently
    rclses = [key for key in priors if 'other' not in key]
    oclses = [key for key in priors if 'other' in key]
    rpriors = np.zeros((len(rclses), len(weights)))
    for i, key in enumerate(rclses): rpriors[i, :] = priors[key]/psum*weights

    # Get the remaining probability and add to others
    opriorsum = np.sum([priors[key]/psum for key in oclses])
    divvy = 1.0 - opriorsum - np.sum(rpriors, axis=0)
    opriors = np.zeros((len(oclses), len(weights)))
    for i, key in enumerate(oclses): opriors[i, :] = (priors[key]/psum)*(1.0 + 1.0/opriorsum*divvy)

    # Put into dict and return
    out = {}
    for i, key in enumerate(rclses): out[key] = rpriors[i, :]
    for i, key in enumerate(oclses): out[key] = opriors[i, :]
    return out

def temporal_prior_weight(traces, actmn, actvar, outliers, fwhm, thresh, priors, expand=1):
    """
    Generate temporal-dependent priors using basis sets and mexican-hat functions
    :param traces: matrix of traces, ncells by nframes
    :param actmn: mean activity
    :param actvar: variation above which we will consider it a guaranteed event
    :param thresh: threshold at which to binarize temporal prior
    :param priors: temporally-independent priors which will be combined with
    :param expand: use a moving max across expand frames
    :return: prior vectors
    """

    # Set the half-width of the convolution kernel
    xhalfwidth = 100

    # Determine a normal function sigma from the full-width at half-maximum
    def sigma(fwhm_): return fwhm_/(2*np.sqrt(2*np.log(2)))

    # Generate the basis functions and correct population activity for baseline and variation
    basis = np.power(fwhm, np.arange(4) + 1)
    popact = (np.nanmean(traces[np.invert(outliers), :], axis=0) - actmn)/actvar
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
        bn[xhalfwidth - defrange:xhalfwidth + defrange + 1] = norm.pdf(range(-defrange, defrange + 1), loc=0, scale=sigma(basis[b]))
        fits[b-1, :] = np.convolve(popact, b0 - bn, 'same')

    # And return the wfits to the narrowest basis function
    weights = np.clip(np.nanmin(fits, axis=0), 0, 1)
    return weights

# ===============================================================================
class ClassifierTrain:
    """
    A class to pass data to the classifier and return data from it. This
    software suite depends on three graphing classes, a class for
    reading in 2p traces, which will have to be adapted per user, and
    has path lib localized to this class and the trace class.

    Path lib in this class are localized to three places: the
    instantiation of the class, base-path in pars, and _get_pathnames.
    """

    def __init__(self, mouse='', dates='', truns=[], crun=0, pars={}):
        """
        Pass the mouse, date, training days, and comparison run. Pars
        will be applied to the default parameters.
        """

        if len(mouse) > 0: pars['mouse'] = mouse
        if len(dates) > 0: pars['training-date'] = dates
        if isinstance(truns, int) or len(truns) > 0: pars['training-runs'] = truns
        if crun > 0: pars['comparison-run'] = crun

        self._pars = {}
        for p in pars: self._pars[p] = pars[p]

        self._mask_cells = None
        self._trained = False
        self._compared = False
        self._framerate = -1
        self._randomize = ''  # 'cells', 'circshift', 'median', 'cells-circshift', 'first-second-order'

        self._randomization_values = []

    def setrandom(self, rand):
        """
        Turn on randomization if it matches the list, otherwise turn it
        off.
        """

        if rand in ['cells', 'circshift', 'median', 'first-second-order', 'identity', 'nonother']:
            self._randomize = rand
            print('Randomizing with type %s'%(rand))
        else:
            self._randomize = ''


        # ----------------------------------------------------------------------
        # PUBLIC FUNCTIONS

    def _decrease_spike_probability(self, onsets, reduction, replicates):
        """
        Extract subsets of spiking activity with lower probabilities.
        Onsets are of shape nonsets, ncells, nframes.
        """

        out = {}
        for cs in onsets:
            nonsets = np.shape(onsets[cs])[0]
            ncells = np.shape(onsets[cs])[1]
            nframes = np.shape(onsets[cs])[2]

            out[cs] = []

            for i in range(nonsets):
                for j in range(replicates):
                    reducecs = (np.random.random(ncells) <= reduction)
                    newonset = np.copy(onsets[cs][i, :, :])
                    newonset[reducecs, :] = 0
                    out[cs].append(newonset)

        return out

    def train(self, pars={}):
        """
        Train the classifier given parameters saved in object/input.
        """

        for p in pars: self._pars[p] = pars[p]
        runs, dates = self._format_run_date()
        if runs == []:
            print('ERROR: Runs not set. Perhaps metadata must be updated?')
            exit(0)

        # Get the stimulus onsets and binarized traces. This function
        # opens each trace and is the slowest part of this.
        # Returns all traces in shape: nonsets, ncells, nframes.
        self.cses = self.csonsets(self._pars['mouse'], dates, runs, self._pars['training-other-running-runs'],
                                  self._pars['other-running-speed-threshold-cms'], self._pars['other-running-fraction'])

        # Add in running periods of other days

        for cs in self.cses:
            self.cses[cs] *= self._pars['analog-training-multiplier']
            self.cses[cs][self.cses[cs] > 1] = 1

        # Generate and save the model
        priors = {cs:self._pars['probability'][cs] for cs in self.cses}

        self.gm = aode.classify(self.cses, priors, self._pars['classification-frames'])

        if self._pars['classifier'] == 'naive-bayes':
            self.gm.generatenb()
        else:
            self.gm.generate()

        # Pass paired CSes if necessary
        # if self._pars['add-class-pairs']:
        # 	print 'ERROR: Have not added this ability yet'
            # dispcses = ['plus', 'neutral', 'minus']
            # for i, cs1 in enumerate(dispcses[:-1]):
            # 	for cs2 in dispcses[i + 1:]:
            # 		if cs1 + '-' + cs2 in self._pars['probability']:
            # 			self.gm.paironset(cs1, cs2, self._pars['probability'][cs1 + '-' + cs2])

        self._trained = True
        return self.gm

    def compare(self, actmn=-1, actvar=-1, actouts=-1):
        """
        STEP 2: Compare a run to trained data using the generative
        model. Save the comparison run for future comparisons.
        """

        import train_classifier, classify_reactivations

        self.cr = paths.gett2p(self._pars['mouse'], self._pars['comparison-date'], self._pars['comparison-run'])
        self.crtr = self.cr.trace(self._pars['trace-type'])[self._mask_cells, :]

        # Deal with randomizing the data
        if len(self._randomize) > 0:
            # self.crtr = self._cut_by_pupil(self.crtr)
            self.crtr = self._cut_by_inactivity(self.crtr)

            # Then, randomize
            if 'cells' in self._randomize or 'identity' in self._randomize:
                crnames = np.arange(np.shape(self.crtr)[0])
                crnames = np.random.choice(crnames, len(crnames), replace=False)
                self.crtr = self.crtr[crnames, :]
                self._randomization_values = crnames
            elif self._randomize == 'nonother':
                crnames = np.arange(np.shape(self.crtr)[0])
                switchable = self.minfocutoff()
                rswitch = np.random.choice(switchable, len(switchable), replace=False)
                crnames[switchable] = rswitch
                self.crtr = self.crtr[crnames, :]
                self._randomization_values = crnames
            else:  # self._randomize == 'circshift':
                self._randomization_values = np.zeros(np.shape(self.crtr)[0], dtype=np.int32)
                for i in range(np.shape(self.crtr)[0]):
                    self._randomization_values[i] = np.random.choice(np.shape(self.crtr)[1])
                    self.crtr[i, :] = np.roll(self.crtr[i, :], self._randomization_values[i])

        self.gm._frames_integrated = self._pars['classification-frames']

        # Set the priors to be vectors
        floatpriors = {cs: self._pars['probability'][cs] for cs in self.cses}
        self._tpriors = floatpriors
        if self._pars['temporal-dependent-priors']:
            fwhmframes = int(round(self._pars['temporal-prior-fwhm-ms']/1000.0*self.cr.framerate))

            priors = temporal_prior(self.crtr, actmn, actvar, actouts[self._mask_cells],
                                    fwhmframes,
                                    self._pars['temporal-prior-threshold'], floatpriors)

            # tpriorvec = classify_reactivations.temporal_prior(self.crtr, actmn, actvar, fwhmframes, actouts)
            # tp2 = classify_reactivations.assign_temporal_priors(floatpriors, tpriorvec, 'other')

            self._tpriors = deepcopy(priors)
        else:
            priors = self._vector_priors(floatpriors, np.shape(self.crtr)[1])

        self._tpriorsbefore = deepcopy(priors)

        # Binarize if desired
        # rc = classify_reactivations.classifier()
        # rc.train(deepcopy(self.cses), ctype='aode')
        # print self._pars['classification-frames'], self._pars[
        #     'analog-comparison-multiplier'], fwhmframes, actmn, actvar, actouts
        # res2 = rc.compare(np.copy(self.crtr), deepcopy(floatpriors), self._pars['classification-frames'], self._pars[
        #     'analog-comparison-multiplier'], fwhmframes, actmn, actvar, actouts)

        self.crtr = np.clip(self.crtr*self._pars['analog-comparison-multiplier'], 0.0, 1.0)

        if self._pars['classifier'] == 'naive-bayes':
            self.results, _, self.likelihoods = \
                self.gm.naivebayes(self.crtr, priors)
        else:
            self.results, _, self.likelihoods = \
                self.gm.compare(self.crtr, priors)

        self._compared = True
        return self.results

    def minfocutoff(self, thresh=0.01, cses=['plus', 'neutral', 'minus']):
        """
        Restrict to only those cells with mutual information > thresh and whose mutual information is highest in the
        cses categories
        :param thresh: mutual information threshold
        :param cses: allowed categories with maximum mutual information
        :return:
        """

        marg = self.gm._marg
        classes = [cl for cl in marg]
        ncells = np.shape(marg[classes[0]])[0]

        maxminfo, maxnonminfo = np.zeros(ncells), np.zeros(ncells)
        for cs in classes:
            minfo = self.mutualinformation(marg, cs)
            if cs in cses:
                maxminfo[minfo > maxminfo] = minfo[minfo > maxminfo]
            else:
                maxnonminfo[minfo > maxnonminfo] = minfo[minfo > maxnonminfo]

        return np.nonzero((maxminfo > thresh) & (maxminfo > maxnonminfo))[0]

    def mutualinformation(self, marg, cs):
        """
        Get the mutual information specific to the marginal, i.e. in the style of Naive-Bayes,
        for a single stimulus
        :param run:
        :param cs:
        :return:
        """

        csprior = 0.5
        noncsprior = 1.0 - csprior;

        classes = [cl for cl in marg]
        ncells = np.shape(marg[classes[0]])[0]
        noncs = [cl for cl in classes if cl != cs]

        out = np.zeros(ncells)

        for c in range(ncells):
            # Sum over states
            for tf in range(2):
                # P(tf|n union m union o union o-running)
                ptfnoncs = 0  # Probability of state tf for non-cs stimuli
                for cl in noncs: ptfnoncs += marg[cl][c, tf]*(1.0/len(noncs))*noncsprior
                ptfnoncs = ptfnoncs/noncsprior
                ptfcs = marg[cs][c, tf]
                ptfsum = ptfnoncs*noncsprior + ptfcs*csprior
                out[c] += ptfcs*csprior*math.log(ptfcs/ptfsum, 2) + ptfnoncs*noncsprior*math.log(ptfnoncs/ptfsum, 2)

        return out

    def _vector_priors(self, priors, nframes):
        """
        Convert scalar priors into vectors of fixed scalar priors.
        :param priors:
        :return:
        """

        psum = 0
        for key in priors: psum += priors[key]
        out = {}
        for key in priors:
            out[key] = np.zeros(nframes) + priors[key]/psum
        return out

    def _cut_by_pupil(self, bindata):
        """
        Restrict by pupil time for randomization.
        """

        # Kick it back if no restriction necessary
        # if (not self._pars['restrict-pupil-diameter'] and not
        # 	self._pars['restrict-pupil-phase']): return bindata

        mask = self.cr.pupilmask(False)
        return bindata[:, mask]

    def _cut_by_inactivity(self, bindata):
        """
        Restrict by times of inactivity for randomization.
        """

        # Kick it back if no restriction necessary
        # if (not self._pars['restrict-pupil-diameter'] and not
        # 	self._pars['restrict-pupil-phase']): return bindata

        mask = self.cr.inactivity()
        return bindata[:, mask]

    def _randomize_with_first_second_order(self, bindata):
        """
        Create a randomized data set that keeps the firing rate per cell
        correct and also the probability of joint firing.
        """

        out = np.zeros(np.shape(bindata))
        rates = np.sum(bindata, 1)
        fframes = np.sum(bindata, 0)  # frames with firing
        onframes = np.arange(len(fframes))[fframes > 0]
        # cells = np.arange(len(rates))

        """
        for f in onframes:
            print f, np.sum(rates), fframes[f]
            active = np.random.choice(cells[rates > 0], fframes[f], False)
            rates[active] -= 1
            out[active, f] = 1


        """
        relrates = []
        for i, r in enumerate(rates):
            for j in range(r):
                relrates.append(i)
        relrates = np.array(relrates)

        for f in onframes:
            active = np.random.choice(relrates, fframes[f], False)
            out[active, f] = 1

        return out

    def _average_firing_rate(self, firing, averaging='median'):
        """
        Randomize the firing of cells with the mean or median firing
        rate.
        """

        out = np.copy(firing)
        rates = np.sum(firing, 1)

        if averaging == 'median':
            av = float(np.median(rates))/np.shape(out)[1]
        else:
            av = float(np.mean(rates))/np.shape(out)[1]
        for i in range(np.shape(out)[0]):
            out[i] = np.random.choice([1, 0], size=(np.shape(out)[1],), p=[av, 1 - av])

        return out

    def describe(self):
        """
        Return data defining the training and comparison days and the
        output of the generative model in the form of a string.
        """
        p = self._pars
        out = '%5s    %6s    %03s    %6s    %03i    %2i  bin %0.2f   %4s %s\n'%(
            p['mouse'], p['training-date'], ','.join([str(i) for i in p['training-runs']]),
            p['comparison-date'], p['comparison-run'], p['stimulus-frames'], -1,
            'rand' if len(self._randomize) > 0 else 'real', self._randomize if
            len(self._randomize) > 0 else '')
        out += '-'*(5 + 4 + 6 + 4 + 3 + 4 + 3 + 4 + 2 + 4 + 4 + 1 + 8 + 9 + 8 + 42) + '\n'
        out += self.gm.describe()
        return out

    def time(self, fr=None, tunits='m'):
        """
        Return the time equivalent of frame fr. If frame is left as
        default, return the x axis range. tunits can be 'm' or 's'
        """

        if not self._compared:
            raise LookupError('Comparison has not yet been run from which we can get timing')

        # If a frame was entered, return a single value
        if fr != None:
            out = fr/self.cr.framerate

            # Fix if minutes
            if tunits == 'm': out = out/60.0
        # If a frame was not entered, return the x axis
        else:
            stims = [key for key in self.results]
            out = (np.arange(len(self.reults[stims[0]]))/self.t.framerate).flatten()

            # Fix if minutes
            if tunits == 'm': out = out/60.0

        return out

    def frame(self, t, tunits='s'):
        """Convert a time to a number of frames."""

        if not self._compared:
            raise LookupError('Comparison has not yet been run from which we can get timing')

        # Convert everything to seconds
        if tunits == 'm': t = t*60.0

        fr = t*self.cr.framerate
        return int(round(fr))

    # ======================================================================
    # ANALYSIS FUNCTIONS

    def pull_results(self):
        """
        Return the classifier results.
        """

        # Check if a comparison has already been run
        if not self._compared: return []

        return self.results

    def pull_likelihoods(self):
        """
        Return the likelihoods from the classifier
        :return: dict of vectors per cs of likelihoods
        """

        if not self._compared: return []
        else: return self.likelihoods

    def pull_marginal(self):
        """
        Return the classifier results.
        """

        return self.gm._marg

    def pull_conditional(self):
        """
        Return the classifier results.
        """

        # Check if a comparison has already been run
        if self._pars['classifier'] == 'naive-bayes': return []

        return self.gm._cond

    def pull_traces(self):
        """
        Return the classifier results.
        """

        # Check if a comparison has already been run
        if not self._compared: return []

        return self.crtr

    def pull_randomizations(self):
        """
        Pull the random numbers used for randomization cell identity or time
        :return:
        """

        if not self._compared: return []
        return self._randomization_values

    def pull_cellmask(self):
        """
        Return a masking array for cells
        :return:
        """

        # Check if a comparison has already been run
        if not self._compared: return []

        return self._mask_cells

    def pull_priors(self):
        """
        Return the vectorized priors
        :return:
        """

        # Check if a comparison has already been run
        if not self._compared: return []

        return self._tpriors

    # ======================================================================
    # LOCAL FUNCTIONS

    def csonsets(self, mouse, date, runs, runruns, runspeed=4.0, runfrac=0.3):
        """
        Open up a series of files specified by mouse, dates, days, and
        pull out all cses. Returns all traces in shape: nonsets, ncells, nframes.
        """

        # Prepare our background values
        cses = {'other': [], 'other-running': self.runonsets(mouse, date, runruns, runspeed)}
        for r in range(len(runs)):
            # Open file with correct trace type and binarize
            t = paths.gett2p(mouse, date, runs[r])

            # Check that the framerate matches the first training run
            self._check_framerate(t, True if r == 0 else False)

            # Get the trace from which to extract time points
            tr = t.trace(self._pars['trace-type'])

            # Search through all stimulus onsets, correctly coding them
            for ncs in t.cses():  # t.cses(self._pars['add-ensure-quinine']):
                fnd = False
                if (ncs in self._pars['training-equivalent'] and
                            len(self._pars['training-equivalent'][ncs]) > 0):
                    cs = self._pars['training-equivalent'][ncs]
                    fnd = True
                elif ncs in self._pars['probability']:
                    cs = ncs
                    fnd = True

                if fnd:
                    if cs not in cses:
                        cses[cs] = []
                    ons = t.csonsets(ncs, 0 if self._pars['train-only-on-positives'] else -1,
                                     self._pars['lick-cutoff'], self._pars['lick-window'])

                    for on in ons:
                        strt = on + self._pars['stimulus-offset-frames']
                        toappend = tr[:, strt:strt + self._pars['stimulus-frames']]
                        if np.shape(toappend)[1] == self._pars['stimulus-frames']:
                            cses[cs].append(toappend)

            # Add all onsets of "other" frames
            others = t.nocs(self._pars['stimulus-frames'], self._pars['excluded-time-around-onsets-frames'], -1)
            # Counts as running at speeds of 4 cm/s
            running = t.speed() > runspeed
            for ot in others:
                strt = ot + self._pars['stimulus-offset-frames']
                if np.nanmean(running[strt:strt + self._pars['stimulus-frames']]) > runfrac:
                    cses['other-running'].append(tr[:, strt:strt + self._pars['stimulus-frames']])
                else:
                    cses['other'].append(tr[:, strt:strt + self._pars['stimulus-frames']])

        # Selectively remove onsets if necessary
        if self._pars['maximum-cs-onsets'] > 0:
            for cs in cses:
                if 'other' not in cs:
                    print('WARNING: Have not yet checked new timing version')
                    # Account for shape of array
                    if len(cses[cs]) > self._pars['maximum-cs-onsets']:
                        cses[cs] = np.random.choice(cses[cs], self._pars['maximum-cs-onsets'], replace=False)

        for cs in cses: cses[cs] = np.array(cses[cs])
        cses = self._prune_nan_cells(cses)
        return cses

    def _prune_nan_cells(self, onsets):
        """
        Remove cells that are NaNs for classification.
        :param onsets: dict of cs onsets of shape nonsets, ncells, nframes.
        :return: dict of cs onsets with cells removed
        """

        # Find all cells with NaNs
        nancells = []
        ncells = -1
        for cs in onsets:
            if len(onsets[cs]) > 0:
                ncells = np.shape(onsets[cs])[1]
                ns = np.sum(np.sum(np.invert(np.isfinite(onsets[cs])), axis=2), axis=0)
                vals = np.arange(ncells)
                nancells.extend(vals[ns > 0])

        # Set _mask_cells if it hasn't been set
        if self._mask_cells is None: self._mask_cells = np.zeros(ncells) < 1

        # Convert nancells to a list of good cells
        nancells = np.array(list(set(nancells)))
        if len(nancells) > 0:
            print('Warning: %i cells have NaNs'%len(nancells))
            self._mask_cells[nancells] = False

        # Copy over only good cells
        out = {}
        for cs in onsets:
            if len(onsets[cs]) > 0:
                out[cs] = np.copy(onsets[cs])[:, self._mask_cells, :]
        return out

    def runonsets(self, mouse, date, runs, runspeed=4):
        """
        Open up files specified by mouse, date, days, and extract all
        running onsets.
        :return: An array of timing onsets nonsets, ncells, nframes.
        """

        out = []
        for i in range(len(runs)):
            # Open file with correct trace type
            t = paths.gett2p(mouse, date, runs[i])

            # Check that the framerate matches the first training run
            self._check_framerate(t, True if i == 0 else False)

            # Get the trace of the correct typeZ
            tr = t.trace(self._pars['trace-type'])

            # Add all onsets of "other" frames
            others = t.nocs(self._pars['stimulus-frames'], self._pars['excluded-time-around-onsets-frames'], runspeed)
            for ot in others:
                strt = ot + self._pars['stimulus-offset-frames']
                out.append(tr[:, strt:strt + self._pars['stimulus-frames']])
        #
        # out = np.array(out)
        # if len(out) > 0:
        # 	out = out[:, self._mask_cells, :]
        return out

    def _check_framerate(self, t2p, accept_change=True):
        """
        Double-check that the framerate of the file matches. If not,
        warn. In the future, add the ability to downsample to match data
        taken with different 2p settings. t2p is a trace2p class,
        accept_change selects whether a change to framerate is allowed.
        """

        if t2p.framerate != self._framerate:
            if accept_change:
                self._framerate = t2p.framerate
                self._convert_time_to_frames()
            else:
                print('ERROR: framerates do not match. Solve.')
                exit(0)
        elif 'stimulus-frames' not in self._pars:
            self._convert_time_to_frames()

        # ----------------------------------------------------------------------
        # ORGANIZATIONAL FUNCTIONS

    def _convert_time_to_frames(self):
        """
        Convert milliseconds to numbers of frames based on the framerate
        """
        self._pars['stimulus-frames'] = int(round(
            self._pars['stimulus-training-ms']/1000.0*self._framerate))
        self._pars['stimulus-offset-frames'] = int(round(
            self._pars['stimulus-training-offset-ms']/1000.0*self._framerate))
        self._pars['classification-frames'] = int(round(
            self._pars['classification-ms']/1000.0*self._framerate))

        # Make the exclusion time a tuple rather than an int
        if isinstance(self._pars['excluded-time-around-onsets-ms'], int):
            self._pars['excluded-time-around-onsets-ms'] = (
                self._pars['excluded-time-around-onsets-ms'],
                self._pars['excluded-time-around-onsets-ms'])
        # Then convert to frames
        self._pars['excluded-time-around-onsets-frames'] = (int(round(
            self._pars['excluded-time-around-onsets-ms'][0]/1000.0*self._framerate)), int(round(
            self._pars['excluded-time-around-onsets-ms'][1]/1000.0*self._framerate)))

    def _format_run_date(self):
        """
        Conver days and dates into lists from optional integers and
        strings. This is important if there are multiple training days.
        """

        runs = self._pars['training-runs']
        dates = self._pars['training-date']
        if isinstance(runs, int): runs = [runs]
        self._pars['training-runs'] = runs
        return (runs, dates)

    def _check_across_day(self, bin, type):
        """
        Fix the ROIs for an across-day comparison.
        """

        # The simple form-- we could check for nonmatching days, but we
        # allow the user to enter a comparison run later
        cd = self._pars['cross-day']
        if len(cd) == 0:
            return bin

        # Initialize the output
        col = 0 if type == 'train' else 1
        obin = []

        # Pull out the appropriate ROIs
        for i in range(np.shape(cd)[1]):
            obin.append(bin[cd[col, i], :])

        return np.array(obin)

    # ===================================================================================

def matlabifypars(pars):
    """
    Convert the parameters into a Matlab-readable version in case
    anyone wants to view them with Matlab.
    """

    out = {}
    for p in pars:
        mlname = p[:31].replace('-', '_')
        if isinstance(pars[p], dict):
            mldict = {}
            for pp in pars[p]:
                mlname = pp[:31].replace('-', '_')
                mldict[mlname] = pars[p][pp]
            out[mlname] = mldict
        else:
            out[mlname] = pars[p]
    return out

def activity(pars):
    """
    Get activity levels for temporal classifier.
    :param pars: parameters from the settings dict
    :return: baseline activity, variance of activity
    """
    from pool import database
    # Set up temporal comparison. This is more complicated than it needs to be so that we don't recalculate the mean
    # activity every time
    actbl, actvar, actouts = -1, -1, -1
    if pars['temporal-dependent-priors']:
        andb = database.db()
        if pars['trace-type'] != 'deconvolved':
            print('ERROR: temporal classifier only implemented for deconvolved data')
            exit(0)

        extramd = metadata.data(pars['mouse'], pars['comparison-date'])
        if 'sated' in extramd and pars['comparison-run'] in extramd['sated']:
            atype = '-sated'
        elif 'hungry' in extramd and pars['comparison-run'] in extramd['hungry']:
            atype = '-hungry'
        else:
            atype = ''

        actbl = andb.get(
            'activity%s-median' % atype, mouse=pars['mouse'],
            date=pars['comparison-date'])
        actouts = andb.get(
            'activity%s-outliers' % atype, mouse=pars['mouse'],
            date=pars['comparison-date'])
        if actbl is None:
            actbl, actvar = 0.01, 0.08 * pars['temporal-prior-baseline-sigma']
        else:
            actbl = actbl * pars['temporal-prior-baseline-activity']
            actvar = andb.get(
                'activity%s-deviation' % atype, mouse=pars['mouse'],
                date=pars['comparison-date']) * \
                pars['temporal-prior-baseline-sigma']

    return actbl, actvar, actouts

def multiclassify(pars, randomize, randomizations=10, verbose=True):
    # Add extra parameters to arguments for classifier

    # Generate the classifier
    cm = ClassifierTrain(pars=pars)

    # Set the randomization parameters
    cm.setrandom(randomize)

    # Train the classifier and return
    cm.train()
    if verbose: print(cm.describe())

    # And run
    actbl, actvar, actouts = activity(pars)

    for i in range(randomizations):
        cm.compare(actbl, actvar, actouts)

        # Prepare output
        out = {
            # 'conditional': cm.pull_conditional(),
            'results': cm.pull_results(),
            'likelihood': cm.pull_likelihoods(),
            'cell_mask': cm.pull_cellmask(),
            'priors': cm.pull_priors(),
        }

        # Save randomized traces if necessary
        # if len(randomize) > 0: out['traces'] = cm.pull_traces()
        if len(randomize) > 0: out['randomization'] = cm.pull_randomizations()
        else: out['marginal'] = cm.pull_marginal(),

        # Get path for parameters and output
        path = paths.output(pars)
        ppath = opath.join(path, 'pars.txt')
        if not opath.exists(ppath): legiblepars.write(ppath, cm._pars)

        # Save the file timestamped (real if not randomized)
        if len(randomize) == 0:
            ts = 'real-{:%y%m%d-%H%M%S}.mat'.format(datetime.datetime.now())
        else:
            ts = 'rand-' + randomize + '-{:%y%m%d-%H%M%S}.mat'.format(datetime.datetime.now())
        path = opath.join(path, ts)

        # Get output replay events for ease of use and save as matlab file
        out['parameters'] = matlabifypars(cm._pars)
        savemat(path, out)

    return out

def superclassify(pars, runs):
    """
    Iterate over a bunch of runs to maximize the rate at which we can reclassify
    :param pars:
    :param runs:
    :return:
    """

    # Generate the classifier
    cm = ClassifierTrain(pars=pars)

    # Set the randomization parameters
    randomize = ''
    cm.setrandom('')

    # Train the classifier and return
    cm.train()
    print(cm.describe())

    for run in runs:
        pars['comparison-run'] = run
        runandsave(cm, pars, '')

def runandsave(cm, pars, randomize):
    """
    Given a trained classifier, cm, and parameters, pars, run and save.
    :param cm: trained classifier
    :param pars: parameters
    :return: None
    """

    actbl, actvar, actouts = activity(pars)
    cm.compare(actbl, actvar, actouts)

    # Prepare output
    out = {
        # 'conditional': cm.pull_conditional(),
        'results': cm.pull_results(),
        'likelihood': cm.pull_likelihoods(),
        'cell_mask': cm.pull_cellmask(),
        'priors': cm.pull_priors(),
    }

    # Save randomized traces if necessary
    # if len(randomize) > 0: out['traces'] = cm.pull_traces()
    if len(randomize) > 0:
        out['randomization'] = cm.pull_randomizations()
    else:
        out['marginal'] = cm.pull_marginal(),

    # Get path for parameters and output
    path = paths.output(pars)
    ppath = opath.join(path, 'pars.txt')
    if not opath.exists(ppath): legiblepars.write(ppath, cm._pars)

    # Save the file timestamped (real if not randomized)
    if len(randomize) == 0:
        ts = 'real-{:%y%m%d-%H%M%S}.mat'.format(datetime.datetime.now())
    else:
        ts = 'rand-' + randomize + '-{:%y%m%d-%H%M%S}.mat'.format(datetime.datetime.now())
    path = opath.join(path, ts)

    # Get output replay events for ease of use and save as matlab file
    out['parameters'] = matlabifypars(cm._pars)
    savemat(path, out)

def classify(pars, randomize, verbose=True, save=True):
    # Add extra parameters to arguments for classifier

    if randomize.lower() == 'repidentity':
        return randomizations.repidentity(pars)

    # Generate the classifier
    cm = ClassifierTrain(pars=pars)

    # Set the randomization parameters
    cm.setrandom(randomize)

    # Train the classifier and return
    cm.train()
    if verbose: print(cm.describe())

    # And run
    actbl, actvar, actouts = activity(pars)
    cm.compare(actbl, actvar, actouts)

    # Prepare output
    out = {
        # 'conditional': cm.pull_conditional(),
        'results': cm.pull_results(),
        'likelihood': cm.pull_likelihoods(),
        'cell_mask': cm.pull_cellmask(),
        'priors': cm.pull_priors(),
    }

    # Save randomized traces if necessary
    # if len(randomize) > 0: out['traces'] = cm.pull_traces()
    if len(randomize) > 0: out['randomization'] = cm.pull_randomizations()
    else: out['marginal'] = cm.pull_marginal(),

    # Prepare saving if necessary
    if save:
        # Get path for parameters and output
        path = paths.output(pars)
        ppath = opath.join(path, 'pars.txt')
        if not opath.exists(ppath): legiblepars.write(ppath, cm._pars)

        # Save the file timestamped (real if not randomized)
        if len(randomize) == 0:
            ts = 'real-{:%y%m%d-%H%M%S}.mat'.format(datetime.datetime.now())
        else:
            ts = 'rand-' + randomize + '-{:%y%m%d-%H%M%S}.mat'.format(datetime.datetime.now())
        path = opath.join(path, ts)

        # Get output replay events for ease of use and save as matlab file
        out['parameters'] = matlabifypars(cm._pars)
        savemat(path, out)
    else:
        out['parameters'] = deepcopy(cm._pars)

    return out


if __name__ == '__main__':
    pass