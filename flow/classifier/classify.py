from copy import deepcopy
import datetime
import math
import numpy as np
import os.path as opath
from scipy.io import savemat
from scipy.stats import norm

from . import analysis
from . import classify as oldclassify
from . import events
import flow.paths as hardcodedpaths
from . import metadata
from . import outfns
from . import randomizations
from flow.paths import gett2p
from .dep import aode, naivebayes, legiblepars


class ReactivationClassifier(object):
    """
    A class to wrap all classifier-related functions
    """

    def __init__(self, parameters={}, outliers=None):
        """
        Create a ReactivationClassifier instance
        :param parameters: a list of parameters, used for keeping track of what parameters were used to generate the
        classifier
        :param outliers: A boolean array of outlier neurons, which will be masked in all remaining functions
        """

        self.model = None
        self.pars = parameters

        # Save output for reusing in other functions
        self._used_priors = None
        self._model_results = None
        self._rollmaxed_comparison = None
        self._original_comparison = None
        self._integrated_frames = None

        # Outputs specifically saved by limiting events
        self._cses = None
        self._number_of_events = None
        self._eventtype_bounds = None
        self._threshold = None

        if outliers is None:
            self._keep = None
        else:
            self._keep = np.invert(outliers)

    def train(self, data, ctype='aode'):
        """
        Train the classifier using data added by trainingdata function and by train
        :param data: dict of 3d arrays of ncells x nframes x ninstances, REMEMBER TO SCALE
        :param ctype: classifier type string, can be 'aode' or 'naive-bayes'
        :return: self
        """

        if self._keep is not None:
            for cs in data:
                data[cs] = data[cs][:, self._keep, :]

        self.model = train(data, ctype)
        return self

    def compare(self, data, priors, integrate_frames=1, analog_scalefactor=1, fwhm=None, actmn=None, actvar=None, tprior_outliers=None,
                skip_temporal_prior=False, reassigning_data=False, merge_cses=[]):
        """
        Run the classifier on a new session of data
        :param data: 2d array of ncells x nframes, REMEMBER TO SCALE
        :param priors: dict of floats for priors
        :return: comparison, dict of classes of vectors of nframes
        """

        if not skip_temporal_prior:
            tpriorvec = temporal_prior(data, actmn, actvar, fwhm, tprior_outliers)
            self._used_priors = assign_temporal_priors(priors, tpriorvec, 'other')
        else:
            self._used_priors = priors

        if not reassigning_data:
            if self._keep is not None:
                data = data[self._keep, :]

            data = np.clip(data*analog_scalefactor, 0.0, 1.0)
            self._original_comparison = np.copy(data)

        self._integrated_frames = integrate_frames
        self._model_results = classify(self.model, data, self._used_priors, integrate_frames)
        self._rollmaxed_comparison = self.model.frames()

        if len(merge_cses) > 0:
            merge1 = merge_cses[0]
            for merge2 in merge_cses[1:]:
                self._model_results[merge1] += self._model_results[merge2]
                self._model_results.pop(merge2)

        return self._model_results

    def limit_to_found_events(self, threshold, inactivity=None, found_events=None, keyword='other'):
        """
        Change the data such that we are only looking at found events.
        :return:
        """

        ncells = np.shape(self._original_comparison)[0]
        self._cses = [cs for cs in self._model_results if keyword not in cs]
        self._number_of_events = {}
        self._eventtype_bounds = {}
        self._threshold = threshold

        tot, newtrace, newpriors = 0, None, {csp: [] for csp in self._used_priors}
        for i, cs in enumerate(self._cses):
            if found_events is not None:
                evs = found_events[cs]
            else:
                evs = np.array(events.peaks(self._model_results[cs], self._original_comparison, threshold))
                evs = np.array([e for e in evs if inactivity[e]])

            self._eventtype_bounds[cs] = (tot, tot + len(evs))
            self._number_of_events[cs] = len(evs)
            tot += len(evs)

            if len(evs) > 0:
                ctrace = self._rollmaxed_comparison[:, evs]
                for csp in self._used_priors:
                    newpriors[csp].extend(self._used_priors[csp][evs].tolist())

                if newtrace is None:
                    newtrace = ctrace
                else:
                    newtrace = np.concatenate([newtrace, ctrace], axis=1)

        for csp in newpriors:
            newpriors[csp] = np.array(newpriors[csp])

        self._original_comparison = newtrace
        self._used_priors = newpriors
        self._integrated_frames = 1
        self.model._frames_integrated = 1

        return np.sum(self._number_of_events.values())

    def dropclassify(self, drop, nrand=-1):
        """
        Drop cells in order specified by drop and reclassify
        :return:
        """

        # Copy the old data so that it can be replaced
        origcompared = np.copy(self._original_comparison)
        origupriors = deepcopy(self._used_priors)
        origmarg, origcond = self.model.learned()
        origmarg, origcond = deepcopy(origmarg), deepcopy(origcond)

        # Get the order in which to drop values
        drop = fixsort(drop, self._keep)
        iterations = len(drop.values()[0])
        counts = {cs: np.array([float(self._number_of_events[cs])] + [0.0]*iterations) for cs in self._cses}

        if nrand < 1:
            # Iterate over the values to drop, beginning with dropping 0
            for i in range(0, iterations+1):
                for cs in self._cses:
                    if self._number_of_events[cs] > 0:
                        usecells = np.zeros(np.sum(self._keep)) < 1
                        usecells[drop[cs][:i]] = False

                        usetimes = np.zeros(np.shape(origcompared)[1]) > 1
                        usetimes[self._eventtype_bounds[cs][0]:self._eventtype_bounds[cs][1]] = True
                        tprior = {cs2: np.copy(origupriors[cs2][usetimes]) for cs2 in origupriors}

                        marg, cond = deepcopy(origmarg), deepcopy(origcond)
                        for cs2 in marg:
                            marg[cs2] = marg[cs2][usecells, :]
                            cond[cs2] = cond[cs2][usecells, :, :]
                            cond[cs2] = cond[cs2][:, usecells, :]
                        self.model.learned(marg, cond)

                        trs = np.copy(origcompared[usecells, :])
                        trs = trs[:, usetimes]
                        res = self.compare(trs, tprior, 1, skip_temporal_prior=True, reassigning_data=True)
                        counts[cs][i] = np.sum(res[cs] > self._threshold)
        else:
            # Iterate over the values to drop, beginning with dropping 0
            for i in range(1, iterations + 1):
                usecells = np.zeros(np.sum(self._keep)) < 1

                for cs in self._cses:
                    usecells[drop[cs][:i]] = False

                marg, cond = deepcopy(origmarg), deepcopy(origcond)
                for cs2 in marg:
                    marg[cs2] = marg[cs2][usecells, :]
                    cond[cs2] = cond[cs2][usecells, :, :]
                    cond[cs2] = cond[cs2][:, usecells, :]
                self.model.learned(marg, cond)

                for cs in self._cses:
                    usetimes = np.zeros(np.shape(origcompared)[1]) > 1
                    usetimes[self._eventtype_bounds[cs][0]:self._eventtype_bounds[cs][1]] = True
                    tprior = {cs2: np.repeat(origupriors[cs2][usetimes], nrand) for cs2 in origupriors}

                    trs = np.copy(origcompared[usecells, :])
                    trs = trs[:, usetimes]
                    trs = np.repeat(trs, nrand, axis=1)
                    for j in range(np.shape(trs)[1]):
                        ranvals = np.random.choice(np.shape(trs)[0], replace=False)
                        trs[:, j] = trs[ranvals, j]

                    res = self.compare(trs, tprior, 1, skip_temporal_prior=True, reassigning_data=True)
                    counts[cs][i] = np.sum(res[cs] > self._threshold)

                # If random, dropcells = np.concatenate([drop[cs][:i] for cs in drop])


            # self.temporal_prior = deepcopy(origupriors)
            # self.model._ncells = np.sum(usecells)
            # for cs in self.model._marg:
            #     self.model._marg[cs] = np.copy(origmarg[cs])[usecells, :]
            #     self.model._cond[cs] = np.copy(origcond[cs])[usecells, :, :]
            #     self.model._cond[cs] = self.model._cond[cs][:, usecells, :]
            #
            # self.compared = np.copy(self.rollmaxed)[usecells, :]
            #
            # if nrand < 1 or i == 0:
            #     res = self.compare()
            #     runtot = 0
            #     for cs in self.cses:
            #         counts[cs][i] = np.sum(res[cs][runtot:self.nevs[cs]] > self.threshold)
            #         runtot += self.nevs[cs]
            # else:
            #     for cs in self.temporal_prior:
            #         self.temporal_prior[cs] = np.repeat(self.temporal_prior[cs], nrand)
            #
            #     self.compared = np.repeat(self.compared, nrand, axis=1)
            #     for j in range(np.shape(self.compared)[1]):
            #         ranvals = np.random.choice(np.shape(self.compared)[0], replace=False)
            #         self.compared[:, j] = self.compared[ranvals, j]
            #
            #     res = self.compare()
            #     runtot = 0
            #     for r in range(nrand):
            #         for cs in self.cses:
            #             if self.nevs[cs] > 0:
            #                 counts[cs][i] += np.sum(np.nanmax([res[cs2][runtot:self.nevs[cs]] for cs2 in self.cses],
            #                                                   axis=0) > self.threshold)/float(nrand)
            #                 runtot += self.nevs[cs]

        # Reset necessary values
        self._original_comparison = origcompared
        self._used_priors = origupriors
        self.model.learned(origmarg, origcond)

        return counts

    def save(self, path=''):
        """
        If a path is passed, save to a .mat file. Otherwise, return as dict
        :param path: path to save to
        :return:
        """

        out = {
            # 'conditional': cm.pull_conditional(),
            'results': self.results(),
            'likelihood': self.likelihoods(),
            'priors': self.temporal_prior,
        }

        # Save randomized traces if necessary
        if len(randomize) > 0:
            out['randomization'] = [],  # cm.pull_randomizations()
        else:
            out['marginal'] = self.model._marg,

        # Prepare saving if necessary
        if len(path) > 0:
            # Get output replay events for ease of use and save as matlab file
            out['parameters'] = matlabifypars(self.pars)
            savemat(path, out)
        else:
            out['parameters'] = deepcopy(self.pars)

        return out

def classifier(parameters={}, outliers=None):
    """
    Create a reactivation classifier instance
    :return: ReactivationClassifier instance
    """

    out = ReactivationClassifier(parameters, outliers)
    return out

def temporal_prior(traces, actmn, actvar, fwhm, outliers=None):
    """
    Generate temporal-dependent priors using basis sets and mexican-hat functions
    :param traces: matrix of traces, ncells by nframes
    :param actmn: mean activity
    :param actvar: variation above which we will consider it a guaranteed event
    :param fwhm: the full-width at half-maximum to use for the temporal prior
    :param outliers: cells with strongly outlying activity
    :return: prior vector
    """

    if outliers is None:
        outliers = np.zeros(np.shape(traces)[0]) > 1

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
    psum = float(np.sum([priors[key] for key in priors]))
    priors = {key:priors[key]/psum for key in priors}
    psum = float(np.sum([priors[key] for key in priors]))

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

def train(tevents, ctype='aode'):
    """
    Train a classifier using a dictionary of events, a dictionary of priors, and a classifier type
    :param tevents: dict of training events of shape ncells x ntimes x nevents
    :param ctype: classifier type, 'aode' or 'naive-bayes'
    :return: a trained classifier model
    """

    gm = None
    if ctype == 'naive-bayes':
        print('naive bayes not implemented yet')
        exit(0)
        gm = naivebayes.classify(tevents, {}, -1)
    else:
        gm = aode.classify(tevents, {}, -1)

    gm.generate()
    return gm

def classify(model, traces, priors, fintegrate, include_likelihoods=False):
    """
    Classify
    :param model: the trained classifier model, from train()
    :param traces: a matrix of ncells x ntimes
    :param priors: a dict of priors, either of floats or of vectors
    :param fintegrate: number of frames to integrate
    :param include_likelihoods: if True, return results, likelihoods
    :return: dict of probabilities at each time point of each category
    """

    nframes = np.shape(traces)[1]
    priors = _sanitize_priors(priors, nframes)

    model._frames_integrated = fintegrate

    results, data, likelihoods = model.compare(traces, priors)

    # if include_likelihoods:
    #     return results, likelihoods, tpriors
    # else:
    #     return results

    return results


def mask(traces, outliers):
    """
    Remove all outlier cells and cells with nans
    :param traces: matrix of ncells x ntimes
    :param outliers: boolean mask of outlier cells
    :return: matrix of ncells (with no outliers) x ntimes
    """

    nancells = np.sum(np.invert(np.isfinite(traces)), axis=1)
    outliers = np.bitwise_or(nancells, outliers)
    return traces[np.invert(outliers), :]

def _sanitize_priors(priors, nframes):
    """
    Make sure that the priors are vectors and that they sum to 1 at each position
    :param priors: dict of priors
    :param nframes: number of frames of the movie, in case the priors are single values
    :return: vector priors summing to 1 at each position
    """

    psum = np.zeros(nframes)
    out = {}

    for key in priors:
        if isinstance(priors[key], float):
            out[key] = np.zeros(nframes) + priors[key]
        else:
            out[key] = priors[key]
        psum += out[key]

    for key in out:
        out[key] = out[key]/psum

    return out

def fixsort(sortdict, keptcells):
    """
    Correct a dict to account for keptcells
    :param sortdict: sorted dictionary of cells to remove
    :param keptcells: boolean numpy array of cells that were kept
    :return: sorteddict with excluded cells removed
    """

    excluded = np.arange(len(keptcells))[np.invert(keptcells)]
    out = {}
    mx = len(sortdict[sortdict.keys()[0]])
    for cs in sortdict:
        out[cs] = np.copy(sortdict[cs])
        for val in excluded:
            # Remove excluded cell if it exists
            out[cs] = out[cs][out[cs] != val]

            # Then subtract from values greater
            out[cs][out[cs] > val] = out[cs][out[cs] > val] - 1

        mx = min(len(out[cs]), mx)

    for cs in out:
        out[cs] = out[cs][:mx]

    return out

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


if __name__ == '__main__':
    from sys import argv
    import parseargv

    out = parseargv.parsekv(argv)
    randomize = parseargv.random(argv)
    classify(out, randomize)
