from copy import deepcopy
import numpy as np
import os.path as opath
import yaml

from .. import config, paths
from ..misc import loadmat, matlabifypars, mkdir_p, savemat, timestamp
from ..misc import wordhash
try:
    from .train import train, updated
except ImportError:
    # Won't be able to train without psytrack installed, but should be able
    # to work with saved .psy files fine.
    pass


class PsyTracker(object):
    """PsyTracker."""

    def __init__(self, mouse, pars=None, verbose=False, force=False):
        """Init."""
        self._mouse = mouse

        if pars is None:
            pars = {}
        self._pars = config.params()['psytrack_defaults']
        self._pars.update(pars)
        self._pars_word = None

        self._path = paths.psytrack(mouse.mouse, self.pars_word)

        self._load_or_train(verbose=verbose, force=force)

        self._weight_labels = None

    def __repr__(self):
        """Repr."""
        return "PsyTracker(path={})".format(self.path)

    @property
    def mouse(self):
        """Return the mouse object."""
        return self._mouse

    @property
    def pars(self):
        """Return the parameters for the PsyTracker."""
        return deepcopy(self._pars)

    @property
    def path(self):
        """The path to the saved location of the data."""
        return self._path

    @property
    def data(self):
        """Return the data used to fit the model."""
        return deepcopy(self.d['data'])

    @property
    def fits(self):
        """Return the fit weights for all parameters."""
        return deepcopy(self.d['results']['model_weights'])

    @property
    def inputs(self):
        """Return the input data formatted for the model."""
        from psytrack.helper.helperFunctions import read_input
        return read_input(self.data, self.weights_dict)

    @property
    def weights_dict(self):
        """Return the dictionary of weights that were fit."""
        return self.pars['weights']

    @property
    def weight_labels(self):
        """The names of each fit weight, order matched to results.

        If a label is repeated, the first instance is the closest in time to
        the current trial, and they step back 1 trial from there.

        """
        if self._weight_labels is None:
            labels = []
            for weight in sorted(self.weights_dict.keys()):
                labels += [weight] * self.weights_dict[weight]
            self._weight_labels = labels
        return deepcopy(self._weight_labels)

    @property
    def pars_word(self):
        if self._pars_word is None:
            pars = self.pars
            pars['updated'] = updated
            self._pars_word = wordhash.word(pars, use_new=True)
        return self._pars_word

    def predict(self, data=None):
        """Return predicted lick probability for every trial.

        Parameters
        ----------
        data : np.ndarray, optional
            If not None, the input data to make predictions from. Should be
            (ntrials x nweights), with the order matching weight_labels(). If
            a bias term was fit, the values should be all 1's.

        Returns
        -------
        prediction : np.ndarray
            A length ntrials array of values on (0, 1) that corresponds to the
            predicted probability of licking on each trial.

        """
        if data is None:
            g = self.inputs
        else:
            g = data
        X = np.sum(g.T * self.fits, axis=0)

        return 1 / (1 + np.exp(-X))

    def _load_or_train(self, verbose=False, force=False):
        if not force:
            try:
                self.d = loadmat(self.path)
                found = True
            except IOError:
                found = False
                if verbose:
                    print('No PsyTracker found, re-calculating.')
            else:
                # Matfiles can't store None, so they have to be converted
                # when saved to disk. I think this is the only place it
                # should be necessary.
                if verbose:
                    print('Saved PsyTracker found, loading: ' + self.path)
                if 'missing_trials' in self.d['data'] and \
                        np.isnan(self.d['data']['missing_trials']):
                    self.d['data']['missing_trials'] = None

        if force or not found:
            data, results = train(self.mouse, verbose=verbose, **self.pars)
            self.d = {
                'data': data,
                'pars': self.pars,
                'results': results,
                'updated': updated,
                'timestamp': timestamp()}
            mkdir_p(opath.dirname(self.path))

            yaml_path = '{}_{}'.format(
                opath.splitext(self.path)[0], 'pars.yml')
            if not opath.exists(yaml_path):
                with open(yaml_path, 'wb') as f:
                    yaml.dump(self.pars, f, encoding='utf-8')

            savemat(self.path, matlabifypars(self.d))
