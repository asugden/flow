"""Train the reactivation classifier."""
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean
import numpy as np

from .. import config
from ..metadata.sorters import Date


def train_classifier(
        date, training_runs=None, training_running_runs=None, outliers=None,
        training_date_int=None, **pars):
    """
    Function to prepare all data to train the classifier.

    Parameters
    ----------
    date : Date
        Date that the classifier will be applied to (and usually run on).
    training_runs : RunSorter, optional
        If specified, use these runs to train the classifier. They should match
        the date or training_date_int. If not passed, defaults to all runs
        from the date of run_type == 'training'.
    training_running_runs : RunSorter, optional
        If specified, use these runs to train running period for the
        classifier. They should match the date or training_date_int. If not
        passed, defaults to all runs from the date of run_type == 'running'.
    outliers : list, optional
        Remove outlier cells from the classifier.
    training_date_int : int, optional
        Optionally train on an alternate date.
    **pars
        All other classifier parameters are collect as keyword arguments. These
        will override the default arguments in ..config.default().

    """
    # Allow for training on a different date than the date that we want to
    # classify.
    if training_date_int is None:
        training_date = date
    else:
        training_date = Date(date.mouse, training_date_int)

    # Get default parameters and update with any new ones passed in.
    classifier_parameters = config.default()
    classifier_parameters.update(pars)

    classifier_parameters = _convert_time_to_frames(
        classifier_parameters, training_date.framerate)

    all_cses = {key: key for key in classifier_parameters['probability']
                if 'disengaged' not in key}
    all_cses.update(classifier_parameters['training-equivalent'])

    if training_runs is None:
        training_runs = training_date.runs(run_types=['training'])
    if training_running_runs is None:
        training_running_runs = training_date.runs(run_types=['running'])

    traces = _get_traces(
        training_runs, training_running_runs, all_cses, classifier_parameters)

    # Removed any binarization
    for cs in traces:
        traces[cs] *= classifier_parameters['analog-training-multiplier']
        traces[cs] = np.clip(traces[cs], 0, 1)

    if outliers is None:
        outliers = _nan_cells(traces)
    else:
        outliers = np.bitwise_and(outliers, _nan_cells(traces))

    # CALL THE CLASSIFIER


def _get_traces(runs, running_runs, all_cses, pars):
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
    pars : dict
        All the classifier parameters.

    Returns
    -------
    dict of np.ndarray
        Keys are cs name, value is nonsets x ncells x nframes

    """
    # Prepare our background values
    out = {'other-running': _get_run_onsets(
        runs=running_runs, length_fr=pars['stimulus-frames'],
        pad_fr=pars['excluded-time-around-onsets-frames'],
        running_threshold_cms=pars['other-running-speed-threshold-cms'],
        offset_fr=pars['stimulus-offset-frames'])}

    for run in runs:
        t2p = run.trace2p()

        # Get the trace from which to extract time points
        trs = t2p.trace('deconvolved')

        # Search through all stimulus onsets, correctly coding them
        for ncs in t2p.cses():  # t.cses(self._pars['add-ensure-quinine']):
            if ncs in all_cses:
                # Remap cs name if needed
                cs = all_cses[ncs]
                # Initialize output
                if cs not in out:
                    out[cs] = []

                ons = t2p.csonsets(
                    ncs, 0 if pars['train-only-on-positives'] else -1,
                    pars['lick-cutoff'], pars['lick-window'])

                for on in ons:
                    start = on + pars['stimulus-offset-frames']
                    toappend = trs[:, start:start + pars['stimulus-frames']]
                    # Make sure interval didn't run off the end.
                    if toappend.shape[1] == pars['stimulus-frames']:
                        out[cs].append(toappend)

        # Add all onsets of "other" frames
        others = t2p.nocs(
            pars['stimulus-frames'],
            pars['excluded-time-around-onsets-frames'], -1)

        # Counts as running at speeds of 4 cm/s
        if len(t2p.speed()) > 0:
            running = t2p.speed() > pars['other-running-speed-threshold-cms']
            for ot in others:
                start = ot + pars['stimulus-offset-frames']
                if nanmean(running[start:start + pars['stimulus-frames']]) > \
                        pars['other-running-fraction']:
                    out['other-running'].append(
                        trs[:, start:start + pars['stimulus-frames']])
                else:
                    out['other'].append(
                        trs[:, start:start + pars['stimulus-frames']])

    # Selectively remove onsets if necessary
    if pars['maximum-cs-onsets'] > 0:
        for cs in out:
            if 'other' not in cs:
                print('WARNING: Have not yet checked new timing version')

                # Account for shape of array
                if len(out[cs]) > pars['maximum-cs-onsets']:
                    out[cs] = np.random.choice(
                        out[cs], pars['maximum-cs-onsets'], replace=False)

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
            # NOTE: what is this offset for?
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
