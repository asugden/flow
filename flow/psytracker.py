from copy import deepcopy
import numpy as np
import os.path as opath
import yaml

from . import config, paths
from .misc import loadmat, matlabifypars, mkdir_p, savemat, timestamp

from .psytrack.train import train

VERSION = 0


class PsyTracker(object):
    """PsyTracker."""

    def __init__(self, mouse, pars=None, verbose=False, force=False):
        """Init."""
        self._mouse = mouse

        if pars is None:
            pars = {}
        self._pars = config.params()['psytrack_defaults']
        self._pars.update(pars)

        self._path = paths.psytrack(mouse.mouse, self.pars)

        self._load_or_train(verbose=verbose, force=force)

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

    def __repr__(self):
        """Repr."""
        return "PsyTracker(path={})".format(self.path)

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
                if False or self.d['version'] < 1:  # Disable version checking for now
                    found = False
                    if verbose:
                        print('Old PsyTracker found, re-calculating.')
                else:
                    # Matfiles can't store None, so they have to be converted
                    # when saved to disk. I think this is the only place it
                    # should be necessary.
                    if verbose:
                        print('Saved PsyTracker found, loading: ' + self.path)
                    if 'missing_trials' in self.d['data'] and \
                            np.isnan(self.d['data']['missing_trials']):
                        self.d['data']['missing_trials'] = None
                    # Pop off matlab junk keys
                    for key in ['__header__', '__version__', '__globals__']:
                        self.d.pop(key, None)
        if force or not found:
            data, results = train(
                self.mouse, verbose=verbose, **self.pars)
            self.d = {
                'data': data,
                'pars': self.pars,
                'results': results,
                'version': VERSION,
                'timestamp': timestamp()}
            mkdir_p(opath.dirname(self.path))

            yaml_path = '{}_{}'.format(
                opath.splitext(self.path)[0], 'pars.yml')
            if not opath.exists(yaml_path):
                with open(yaml_path, 'wb') as f:
                    yaml.dump(self.pars, f, encoding='utf-8')

            savemat(self.path, matlabifypars(self.d))
