from .misc import loadmat
from .classifier import randomizations
from .classifier import train
from .classifier.base_classifier import BaseClassifier


class NoClassifierError(Exception):
    pass


class Classify2P(BaseClassifier):
    def __init__(self, path, run, pars=None):
        """Load in a classifier or classifiers.

        Parameters
        ----------
        path : str
            Path to the classifier to load or save if needed.
        run : Run
            The run object that the classifier refers to.
        pars : dict
            The parameters used to generate the classifier.

        """
        BaseClassifier.__init__(self)

        if pars is None:
            pars = {}
        self.pars = pars
        self.run = run
        self._path = path
        self._trained_model = None
        self._trained_params = None
        self._trained_activity = None
        self._trained_nan_cells = None
        self._trained_activity_scale = None

        self.d = None
        self._load_or_classify(path)

    def __repr__(self):
        return "Classify2P(path={})".format(self._path)

    @property
    def frame_range(self):
        """The frames that should be compared due to maxing.

        Left side included, right side excluded.

        """
        t2p = self.run.trace2p()
        integrate_frames = int(round(
            self.pars['classification-ms']/1000.0*t2p.framerate))
        fmin = -int(integrate_frames//2.0)
        fmax = fmin + integrate_frames

        return fmin, fmax

    def train(self):
        """
        Train a model and return it.

        Returns
        -------
        Trained classifier model

        """

        if self._trained_model is None:
            self._trained_model, self._trained_params, \
                self._trained_nan_cells, self._trained_activity_scale = \
                train.train_classifier(run=self.run, **self.pars)

        out = {
            'parameters': self.pars,
            'marginal': model.marginal,
            'conditional': model.conditional,
            'cell_mask': np.invert(self._trained_nan_cells),
        }

        return out

    def classify(
            self, data=None, priors=None, temporal_prior=None,
            integrate_frames=None):
        """
        Return a trained classifier.

        Used either for running the traditional classifier or for
        randomization.

        data : matrix
            Matrix of data to compare, ncells x ntimes
        priors : dict
            The prior probabilities for each class. Defaults to pars.
        temporal_prior : vector
            A vector of weights per unit time. Defaults to the standard
            temporal prior if set in pars.
        integrate_frames : int
            The number of frames to integrate.

        Returns
        -------
        dict
            The standard output format of train.py

        """

        # Train only once
        if self._trained_model is None:
            self._trained_model, self._trained_params, \
                self._trained_nan_cells, self._trained_activity_scale = \
                train.train_classifier(run=self.run, **self.pars)

        results = train.classify_reactivations(
            run=self.run, model=self._trained_model,
            pars=self._trained_params, nan_cells=self._trained_nan_cells,
            activity_scale=self._trained_activity_scale,
            replace_data=data, replace_priors=priors,
            replace_temporal_prior=temporal_prior,
            replace_integrate_frames=integrate_frames)

        return results

    def randomization(self, rtype, **kwargs):
        """
        Return an object of the correct randomization type.

        Parameters
        ----------
        rtype : str {'identity', 'time'}
            Randomization type
        **kwargs
            Additional keyword arguments are passed to the individual
            randomization classes.

        Returns
        -------
        object
            Randomization object

        """

        if rtype == 'identity':
            return randomizations.identity.RandomizeIdentity(self, **kwargs)
        else:
            return randomizations.time.RandomizeTime(self, **kwargs)

    def _load_or_classify(self, path):
        try:
            self.d = loadmat(path)
            found = True
        except IOError:
            found = False

        if not found:
            try:
                self._classify(path)
            except ValueError as err:
                # Skip dates that just can't be trained
                if 'Not enough training data' in str(err):
                    raise NoClassifierError(
                        'Not enough training data: {}'.format(self.run))
                elif 'stimulus_length should only be 0 if there are no cses' in str(err):
                    raise NoClassifierError(
                        "Bad simpcell, can't calculate stimulus_length: {}".format(
                            self.run))
                elif 'traces missing' in str(err):
                    raise NoClassifierError(err)
                else:
                    raise

    def _classify(self, path):
        """Run the classifier and save the results."""
        self.d = self.classify()
        self._save(path)
