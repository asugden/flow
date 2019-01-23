from __future__ import print_function
# Probably needs to be refactored still: 180821
import datetime
import numpy as np
import os.path as opath
from scipy.io import savemat

from .. import classify2p, paths
from ..misc import legiblepars

# Imported within functions that need them
# from . import classify
# from . import parseargv

def repevents(gm, t2p, combframes, reps):
    """
    Find all replay events
    :param gm:
    :param t2p:
    :param combframes
    :param reps:
    :return:
    """

    dec = t2p.trace('deconvolved')
    ncells = np.shape(dec)[0]
    crnames = np.arange(ncells)

    fmin = t2p.lastonset()
    pmask = t2p.inactivity()

    fmin = max(2, fmin)
    cses = [cs for cs in gm['results'] if 'other' not in cs]

    framematch, trace, priors = None, None, None
    for i, cs in enumerate(cses):
        evs = np.array(classify2p.peakprobs(gm['results'][cs], 0.05))
        evs = evs[evs > fmin]
        print((cs, len(evs)))

        ctrace = np.zeros((ncells, len(evs)*(reps + 1)))
        cfmatch = np.zeros((len(evs), 4))  # cs, frame, probability, pupilmask at frame
        if priors is None: priors = {csp:[] for csp in gm['priors']}

        for j, ev in enumerate(evs):
            cfmatch[j, :] = [i, ev, gm['results'][cs][ev], pmask[ev]]
            ctrace[:, j*(reps+1)] = np.nanmax(dec[:, ev:ev+combframes], axis=1)
            for r in range(reps):
                crnames = np.random.choice(crnames, len(crnames), replace=False)
                ctrace[:, j*(reps + 1) + (r + 1)] = ctrace[crnames, j*(reps+1)]

            for csp in gm['priors']:
                priors[csp].extend([gm['priors'][csp][ev - 2]]*(reps + 1))  # The minus 2 is through matching with
                                                                            # AODE... confusing

        # if i == 0:
        #     # ev = evs[0]
        #     ev = 6409
        #     for k in range(10):
        #         for csp in gm['priors']:
        #             priors[csp][k] = gm['priors'][csp][ev+k-5]
        #         ctrace[:, k] = ctrace[:, 0]

        if trace is None:
            trace = ctrace
            framematch = cfmatch
        else:
            trace = np.concatenate([trace, ctrace], axis=1)
            framematch = np.concatenate([framematch, cfmatch], axis=0)

    for csp in priors: priors[csp] = np.array(priors[csp])
    return trace, priors, cses, framematch

def repidentity(pars):
    """
    Randomize replay identify from parameters
    :param pars:
    :return: output of classifier

    """
    from . import parseargv
    gm = parseargv.classifier(pars, randomize='', force=True)
    return randomizeevents(pars, gm)

def randomizeevents(pars, preclassifier, reps=100, verbose=True):
    """
    A special version of randomization that takes individual replay events from a previous classification,
    then randomizes the identity reps times only within those frames.
    :param pars: classifier parameters
    :return:

    """
    from . import classify
    # Generate the classifier
    cm = classify.ClassifierTrain(pars=pars)
    cm.setrandom('')

    # Train the classifier and return
    cm.train()
    if verbose: print(cm.describe())

    t2p = paths.gett2p(pars['mouse'], pars['comparison-date'], pars['comparison-run'])
    searchms = pars['classification-ms']

    trace, priors, cses, framematch = repevents(preclassifier, t2p,
                                                preclassifier['parameters']['classification_frames'], reps)

    # And run
    cm.comparetotrace(trace, priors)

    # Prepare output
    out = {
        'marginal': cm.pull_marginal(),
        'likelihood': cm.pull_likelihoods(),
        'results': cm.pull_results(),
        'cell_mask': cm.pull_cellmask(),
        'priors': cm.pull_priors(),
        'traces': trace,
        'replicates': reps,
        'codes': cses,
        'frame-match': framematch,
    }

    # Get path for parameters and output
    path = paths.output(pars)
    ppath = opath.join(path, 'pars.txt')
    if not opath.exists(ppath): legiblepars.write(ppath, cm._pars)

    # Save the file timestamped (real if not randomized)
    ts = 'rand-repidentity-{:%y%m%d-%H%M%S}.mat'.format(datetime.datetime.now())
    path = opath.join(path, ts)

    # Get output replay events for ease of use and save as matlab file
    out['parameters'] = classify.matlabifypars(cm._pars)
    savemat(path, out)

    return out

if __name__ == '__main__':
    from sys import argv
    from flow import parseargv
    runs = parseargv.sortedruns(argv, classifier=True, trace=False, force=True)
    while runs.next():
        md, args, gm = runs.get()
        vals = parseargv.classifier(args, 'repidentity', True)
        print(vals['frame-match'][0])
        print(vals['priors']['neutral'][0])
        print(vals['results']['neutral'][0:10])
        import pdb;pdb.set_trace()
