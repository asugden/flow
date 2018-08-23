import numpy as np


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
        skipns = np.convolve(flat, [1]+[0]*i+[1], 'same')
        flat[skipns == 2] = 1

    # Event onsets
    ons = np.concatenate([[flat[0]], np.diff(flat)])
    offs = np.concatenate([np.diff(flat), [-1*flat[-1]]])
    ons = np.arange(len(ons))[ons > 0]
    offs = np.arange(len(offs))[offs < 0] + 1  # Before the + 1, the off was the last frame, not the following frame

    # Now we can account for the case in which there is no max set
    if max >= 1.0:
        if offsets: return ons, offs
        else: return ons

    # Account for the maximum
    out = []
    outoffs = []
    for i in range(len(ons)):
        if np.max(result[ons[i]:offs[i]]) < max:
            if all:
                out.extend(range(ons[i], offs[i]))
            else:
                out.append(ons[i])
                outoffs.append(offs[i])

    if not all and offsets: return np.array(out), np.array(outoffs)
    else: return np.array(out)

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

def classpeaks(result, threshold, max=2, downfor=2, maxlen=-1):
    """
    Return the times of peak classification of replay events found by counts

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
            out.append(evs[i] + np.argmax(result[evs[i]:evoffs[i]]))

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

def counts(results, threshold, all=False, max=2):
    """
    Return count for each cs that is not other

    :param results: results dict from classifier, e.g. classifier['results']
    :param threshold: probability threshold above which one should count
    :param all: return all time points if True, otherwise just event onsets
    :param max: maximum allowed
    :return: dict of frame numbers
    """

    # Account for calling mistake, if classifier is directly passed
    if 'results' in results: results = results['results']

    out = {}
    for key in results:
        if 'other' not in key:
            out[key] = count(results[key], threshold, all, max)
    return out
