"""Train the reactivation classifier."""
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean
import numpy as np

from .. import config


def train_classifier(
        run, training_runs=None, running_runs=None, remove_outliers=True,
        training_date=None, **pars):
    """
    Function to prepare all data to train the classifier.

    Parameters
    ----------
    run : Run
        Run that the classifier will be applied to.
    training_runs : RunSorter, optional
        If specified, use these runs to train the classifier. If not passed,
        defaults to all runs from the date of run_type == 'training'.
    running_runs : RunSorter, optional
        If specified, use these runs to train running period for the
        classifier. If not passed, defaults to all runs from the date of
        run_type == 'running'.
    remove_outliers : bool
        Remove outlier cells from the temporal prior.
    training_date : Date, optional
        Optionally train on an alternate date.
    **pars
        All other classifier parameters are collect as keyword arguments. These
        will override the default arguments in ..config.default().

    """
    from replay.lib import classify_reactivations
    # Allow for training on a different date than the date that we want to
    # classify.
    if training_date is None:
        training_date = run.parent
    else:
        # This hasn't really been tested. First person to use it, feel free to
        # change the input to whatever is useful (Date object, relative day
        # shift, absolute date int, etc.).
        if training_date != run.parent:
            raise NotImplementedError

    # Infer training and running runs if they are not specified
    if training_runs is None:
        training_runs = training_date.runs(
            run_types=['training'], tags=['hungry'])
    else:
        assert all(run.parent==training_date for run in training_runs)
    if running_runs is None:
        running_runs = training_date.runs(
            run_types=['running'], tags=['hungry'])
    else:
        assert all(run.parent==training_date for run in running_runs)

    # Get default parameters and update with any new ones passed in.
    params = config.default()
    params.update(pars)

    # Convert parameters specified in seconds into frames.
    params = _convert_time_to_frames(
        params, training_date.framerate)

    # Collect all classifier states
    all_cses = {key: key for key in params['probability']
                if 'disengaged' not in key}
    all_cses.update(params['training-equivalent'])

    # Pull all training data
    traces = _get_traces(
        training_runs, running_runs, all_cses,
        trace_type=params['trace-type'],
        length_fr=params['stimulus-frames'],
        pad_fr=params['excluded-time-around-onsets-frames'],
        offset_fr=params['stimulus-offset-frames'],
        running_threshold_cms=params['other-running-speed-threshold-cms'],
        lick_cutoff=params['lick-cutoff'],
        lick_window=params['lick-window'],
        correct_trials=params['train-only-on-positives'],
        running_fraction=params['other-running-fraction'],
        max_n_onsets=params['maximum-cs-onsets'])

    # Remove any binarization
    for cs in traces:
        traces[cs] *= params['analog-training-multiplier']
        traces[cs] = np.clip(traces[cs], 0, 1)

    # Calculate baseline and variance for each cell, and identify outliers
    baseline, variance, outliers = _activity(
        run,
        baseline_activity=params['temporal-prior-baseline-activity'],
        baseline_sigma=params['temporal-prior-baseline-sigma'],
        trace_type=params['trace-type'])

    # We want to at least mask out cells with NaN's, maybe also outliers.
    if not remove_outliers:
        outliers = _nan_cells(traces)
    else:
        outliers = np.bitwise_or(outliers, _nan_cells(traces))

    ##
    ## All code below this can be replaced with new classifier code
    ##
    priors = {cs: params['probability'][cs] for cs in traces}
    rc = classify_reactivations.ReactivationClassifier(
        params, outliers)
    rc.train(traces, params['classifier'])

    full_traces = run.trace2p().trace(params['trace-type'])

    rc.compare(
        full_traces,
        priors,
        integrate_frames=params['classification-frames'],
        analog_scalefactor=params['analog-comparison-multiplier'],
        fwhm=params['temporal-prior-fwhm-frames'],
        actmn=baseline,
        actvar=variance,
        tprior_outliers=outliers,
        skip_temporal_prior=False,
        reassigning_data=False,
        merge_cses=[])

    params.update({'comparison-date': run.date,
                   'comparison-run': run.run,
                   'mouse': unicode(run.mouse),
                   'training-date': training_date.date,
                   'training-other-running-runs':
                       sorted(r.run for r in running_runs),
                   'training-runs':
                       sorted(r.run for r in training_runs)})

    out = {'parameters': params,
           'results': rc._model_results,
           'likelihood': rc._model_likelihood,
           'marginal': rc.model._marg,
           'priors': rc._used_priors}

    return out


def _activity(
        run, baseline_activity=0., baseline_sigma=3.0,
        trace_type='deconvolved'):
    """
    Get the activity levels for the temporal classifier.

    Parameters
    ----------
    run : Run
    baseline_activity : float
        Scale factor of baseline activity.
        'temporal-prior-baseline-activity' in classifier parameters.
    baseline_sigma : float
        Scale factor of baseline variance.
        'temporal-prior-baseline-sigma' in classifier parameters.
    trace_type : {'deconvolved'}
        This only works on deconvolved data for now.

    Returns
    -------
    baseline activity, variance of activity, outliers

    """
    if trace_type != 'deconvolved':
        raise ValueError('Temporal classifier only implemented for deconvolved data.')

    if 'sated' in run.tags:
        runs = run.parent.runs(run_types=['spontaneous'], tags=['sated'])
        spontaneous = True
    elif 'hungry' in run.tags:
        runs = run.parent.runs(run_types=['spontaneous'], tags=['hungry'])
        spontaneous = True
    else:
        runs = run.parent.runs(run_types=['training'])
        spontaneous = False

    baseline, variance, outliers = None, None, None
    if spontaneous:
        popact, outliers = [], []
        for r in runs:
            t2p = r.trace2p()
            pact = t2p.trace('deconvolved')
            fmin = t2p.lastonset()
            mask = t2p.inactivity()
            mask[:fmin] = False

            if len(popact):
                popact = np.concatenate([popact, pact[:, mask]], axis=1)
            else:
                popact = pact[:, mask]

            trs = t2p.trace('deconvolved')[:, fmin:]
            cellact = np.nanmean(trs, axis=1)
            outs = cellact > np.nanmedian(cellact) + 2*np.std(cellact)

            if len(outliers) == 0:
                outliers = outs
            else:
                outliers = np.bitwise_or(outliers, outs)

        if len(popact):
            popact = np.nanmean(popact[np.invert(outliers), :], axis=0)

            baseline = np.median(popact)
            variance = np.std(popact)
            outliers = outliers
    else:
        popact = []
        for r in runs:
            t2p = r.trace2p()
            ncells = t2p.ncells
            pact = np.nanmean(t2p.trace('deconvolved'), axis=0)
            skipframes = int(t2p.framerate*4)

            for cs in ['plus', 'neutral', 'minus', 'pavlovian']:
                onsets = t2p.csonsets(cs)
                for ons in onsets:
                    pact[ons:ons+skipframes] = np.nan
            popact = np.concatenate([popact, pact[np.isfinite(pact)]])

        if len(popact):
            baseline = np.median(popact)

            # Exclude extremes
            percent = 2.0
            popact = np.sort(popact)
            trim = int(percent*popact.size/100.)
            popact = popact[trim:-trim]

            variance = np.str(popact)
            outliers = np.zeros(ncells, dtype=bool)

    if baseline is None:
        baseline, variance = 0.01, 0.08*baseline_sigma
    else:
        baseline *= baseline_activity
        variance *= baseline_sigma

    return baseline, variance, outliers


def _get_traces(
        runs, running_runs, all_cses, trace_type='deconvolved', length_fr=15,
        pad_fr=31, offset_fr=1, running_threshold_cms=4., correct_trials=False,
        lick_cutoff=-1, lick_window=(-1, 0), running_fraction=0.3,
        max_n_onsets=-1):
    """
    Return all trace data by stimulus chopped into same sized intervals.

    Parameters
    ----------
    runs : RunSorter
        Runs to use for training of stimulus presentations.
    running_runs : RunSorter
        Running-only runs used to train running times.
    all_cses : dict
        Re-map cs names in order to combine some cses.
    length_fr : int
        Chop up the training data into chunks of this many frames.
    pad_fr : int or (int, int)
        Number of frames to pad around the stimulus onset.
    offset_fr : int
        Number of frames to pad around the stimulus offset.
    running_threshold_cms : float
        Minimum threshold above which the mouse is considered to be running.
    correct_trials : bool
        Limit training to only correct trials.
    lick_cutoff : int, optional
        Toss trials with more than lick_cutoff licks.
    lick_window : tuple, optional
        Count licks within this window to choose which to toss out.
    running_fraction : float
        Intervals must have greater than this fraction of frames to be
        considered running.
    max_n_onsets : int, optional
        If >0, limit the number of allowed onsets to this number, to match
        the amount of training data across states.

    Returns
    -------
    dict of np.ndarray
        Keys are cs name, value is nonsets x ncells x nframes

    """
    # Prepare running baseline data.
    # NOTE: running thresholding is done differently here than later during
    # stimulus runs.
    out = {'other-running': _get_run_onsets(
        runs=running_runs,
        length_fr=length_fr,
        pad_fr=pad_fr,
        offset_fr=offset_fr,
        running_threshold_cms=running_threshold_cms)}

    for run in runs:
        t2p = run.trace2p()

        # Get the trace from which to extract time points
        trs = t2p.trace(trace_type)

        # Search through all stimulus onsets, correctly coding them
        for ncs in t2p.cses():  # t.cses(self._pars['add-ensure-quinine']):
            if ncs in all_cses:
                # Remap cs name if needed
                # NOTE: blank trials are just labeled 'other' and not checked
                # for running.
                cs = all_cses[ncs]
                # Initialize output
                if cs not in out:
                    out[cs] = []

                ons = t2p.csonsets(
                    ncs, 0 if correct_trials else -1, lick_cutoff, lick_window)

                for on in ons:
                    start = on + offset_fr
                    toappend = trs[:, start:start + length_fr]
                    # Make sure interval didn't run off the end.
                    if toappend.shape[1] == length_fr:
                        out[cs].append(toappend)

        # Add all onsets of "other" frames
        others = t2p.nocs(length_fr, pad_fr, -1)

        if len(t2p.speed()) > 0:
            running = t2p.speed() > running_threshold_cms
            for ot in others:
                start = ot + offset_fr
                if nanmean(running[start:start + length_fr]) > \
                        running_fraction:
                    out['other-running'].append(
                        trs[:, start:start + length_fr])
                else:
                    out['other'].append(
                        trs[:, start:start + length_fr])

    # Selectively remove onsets if necessary
    if max_n_onsets > 0:
        for cs in out:
            if 'other' not in cs:
                print('WARNING: Have not yet checked new timing version')

                # Account for shape of array
                if len(out[cs]) > max_n_onsets:
                    out[cs] = np.random.choice(
                        out[cs], max_n_onsets, replace=False)

    for cs in out:
        out[cs] = np.array(out[cs])

    return out


def _get_run_onsets(
        runs, length_fr, pad_fr, running_threshold_cms, offset_fr):
    """
    Return times of running without stimulus.

    Parameters
    ----------
    runs : RunSorter
        Runs in which to search for running time.
    length_fr : int
        Desired length of each running interval.
    pad_fr : int
        Number of frames to pad around the stimulus.
    offset_ft : int
        Number of frames to shift running intervals.

    Returns
    -------
    list of np.ndarray
        nonsets x (ncells x nframes)

    """
    out = []
    for run in runs:
        t2p = run.trace2p()
        tr = t2p.trace('deconvolved')

        # Add all onsets of "other" frames
        others = t2p.nocs(length_fr, pad_fr,
                          running_threshold_cms)
        for ot in others:
            start = ot + offset_fr
            out.append(tr[:, start:start + length_fr])

    return out


def _convert_time_to_frames(pars, framerate):
    """Convert milliseconds to numbers of frames based on the framerate."""
    pars['stimulus-frames'] = \
        int(round(pars['stimulus-training-ms']/1000.0*framerate))
    pars['stimulus-offset-frames'] = \
        int(round(pars['stimulus-training-offset-ms']/1000.0*framerate))
    pars['classification-frames'] = \
        int(round(pars['classification-ms']/1000.0*framerate))

    # Make the exclusion time a tuple rather than an int
    if isinstance(pars['excluded-time-around-onsets-ms'], int):
        pars['excluded-time-around-onsets-ms'] = (
            pars['excluded-time-around-onsets-ms'],
            pars['excluded-time-around-onsets-ms'])

    # Then convert to frames
    pars['excluded-time-around-onsets-frames'] = (
        int(round(pars['excluded-time-around-onsets-ms'][0]/1000.0*framerate)),
        int(round(pars['excluded-time-around-onsets-ms'][1]/1000.0*framerate)))

    pars['temporal-prior-fwhm-frames'] = \
        int(round(pars['temporal-prior-fwhm-ms']/1000.0*framerate))

    return pars


def _nan_cells(traces):
    """
    Return mask of cells with any NaN values for any stimulus.

    Parameters
    ----------
    traces : dict of np.ndarray
        Keys are stimulus names, values are nonsets x ncells x nframes arrays.

    Returns
    -------
    np.ndarray
        ncells length mask, True for cells that have any NaN's in their traces.

    """
    # Find all cells with NaNs
    nancells = []
    ncells = -1
    for cs in traces:
        if len(traces[cs]) > 0:
            ncells = np.shape(traces[cs])[1]
            ns = np.sum(np.sum(np.invert(np.isfinite(
                traces[cs])), axis=2), axis=0)
            vals = np.arange(ncells)
            nancells.extend(vals[ns > 0])

    # Set _mask_cells if it hasn't been set
    out = np.zeros(ncells, dtype=bool)

    # Convert nancells to a list of good cells
    nancells = np.array(list(set(nancells)))
    if len(nancells) > 0:
        print('Warning: %i cells have NaNs'%len(nancells))
        out[nancells] = True

    return out
