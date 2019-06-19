"""Object representations of Mouse/Date/Run and list of each.

Used to sort through mouse metadata and is the core of analysis functions.

"""
from __future__ import print_function
from builtins import object
from copy import copy
from datetime import datetime
from future.moves.collections import UserList
import json
import numpy as np
from pandas import IndexSlice as Idx

from .metadata import metadata
from .misc import timestamp
from .psytrack import psytracker
from . import classify2p, config, glm, paths, xday


class Mouse(object):
    """A single Mouse.

    Parameters
    ----------
    mouse : str

    Attributes
    ----------
    mouse : str
    tags : tuple of str

    Methods
    -------
    dates
        Return a DateSorter of associated Dates, as determined by the metadata.

    """

    def __init__(self, mouse):
        """Init."""
        self._mouse = str(mouse)
        self._tags = None

    @property
    def mouse(self):
        """Mouse name as string."""
        return copy(self._mouse)

    @property
    def tags(self):
        """Tuple of mouse tags."""
        if self._tags is None:
            self._get_metadata()
        return tuple(self._tags)

    def _get_metadata(self):
        """Query the metadata and set necessary properties."""
        meta = metadata.meta(mice=[self.mouse])
        mouse_tags = set(meta['tags'].iloc[0])
        for tags in meta['tags']:
            mouse_tags.intersection_update(tags)

        self._tags = tuple(sorted(mouse_tags))

    def __repr__(self):
        """Return repr of Mouse."""
        return "Mouse(mouse='{}', tags={})".format(self.mouse, self.tags)

    def __hash__(self):
        """Hash of a Mouse."""
        return hash(self.__repr__())

    def __str__(self):
        """Return str of Mouse."""
        return self.mouse

    def __lt__(self, other):
        """Less than."""
        if not isinstance(other, type(self)):
            raise NotImplemented
        return self.mouse < other.mouse

    def __le__(self, other):
        """Less than or equal to."""
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other):
        """Greater than."""
        return not self.__le__(other)

    def __ge__(self, other):
        """Greater than or equal to."""
        return not self.__lt__(other)

    def __eq__(self, other):
        """Equal."""
        return isinstance(other, type(self)) and self.mouse == other.mouse

    def __ne__(self, other):
        """Not equal."""
        return not self.__eq__(other)

    def dates(self, dates=None, tags=None, exclude_tags='bad', name=None):
        """Return a DateSorter of associated Dates.

        Can optionally filter Dates by tags.

        Parameters
        ----------
        dates : list of int, optional
            List of dates to include. Can also be a single date.
        tags : list of str, optional
            List of tags to filter on. Can also be a single tag.
        exclude_tags : list of str, optional
            List of tags to exclude. Can also be a single tag.
        name : str, optional
            Name of resulting DateSorter.

        Returns
        -------
        DateSorter

        """
        if name is None:
            name = str(self) + ' dates'

        meta = metadata.meta(
            mice=[self.mouse], dates=dates, tags=tags,
            exclude_tags=exclude_tags)
        meta_dates = meta.index.get_level_values('date').unique()

        date_objs = (Date(mouse=self.mouse, date=date) for date in meta_dates)

        return DateSorter(date_objs, name=name)

    def psytracker(self, newpars=None, verbose=False, force=False):
        """Load or calculate a PsyTracker for this mouse.

        Parameters
        ----------
        newpars : dict, optional
            Override default parameters for the PsyTracker. See
            flow.psytrack.train.train for options.
        verbose : bool
            Be verbose.
        force : bool
            If True, ignore saved PsyTracker and re-calculate.

        """
        return psytracker.PsyTracker(
            self, pars=newpars, verbose=verbose, force=force)


class Date(object):
    """A single Date.

    Parameters
    ----------
    mouse : str
    date : int
    cells : numpy array

    Attributes
    ----------
    mouse : str
    date : int
    cells : numpy array
        A vector of cell numbers, used to reorder trace2p if comparing across
        days
    tags : tuple of str
        Specific tags for this date. Collected from metadata.
    photometry : tuple of str
        Tuple of labels for the location of photometry recordings on this
        date, if present.
    parent : Mouse
        The parent Mouse object.

    Methods
    -------
    runs
        Return a RunSorter of associated Runs, as determined by the metadata.

    """

    def __init__(self, mouse, date, cells=None):
        """Init."""
        self._mouse = str(mouse)
        self._date = int(date)
        self._cells = cells

        self._parent = Mouse(mouse=self.mouse)
        self._tags, self._photometry = None, None
        self._runs = None
        self._framerate = None
        self._glm = {}

    @property
    def mouse(self):
        """Name of mouse as string."""
        return copy(self._mouse)

    @property
    def date(self):
        """Date as integer."""
        return copy(self._date)

    @property
    def parent(self):
        """Link back to the parent Mouse object."""
        return self._parent

    @property
    def tags(self):
        """Tuple of Date tags."""
        if self._tags is None:
            self._get_metadata()
        return tuple(self._tags)

    @property
    def cells(self):
        return copy(self._cells)

    @property
    def framerate(self):
        """Imaging framerate for this date."""
        if self._framerate is None:
            for run in self.runs():
                try:
                    t2p = run.trace2p()
                except IOError:
                    continue
                self._framerate = t2p.framerate
                break
            if self._framerate is None:
                raise ValueError(
                    'Unable to determine framerate for {}'.format(self) +
                    ', no imaging data available.')
        return self._framerate

    def set_subset(self, val=None):
        """
        Set the cell indices to be subset.

        Parameters
        ----------
        val : None or array
            Indices of cells to be kept, or all cells if val is None

        """

        self._cells = val

    @property
    def photometry(self):
        """Tuple of photometry information."""
        if self._photometry is None:
            self._get_metadata()
        return tuple(self._photometry)

    def _get_metadata(self):
        """Query the metadata and set necessary properties."""
        meta = metadata.meta(mice=[self.mouse], dates=[self.date])
        date_tags = set(meta['tags'].iloc[0])
        for tags in meta['tags']:
            date_tags.intersection_update(tags)
        photometry = set(meta['photometry'].iloc[0])
        for photo in meta['photometry']:
            photometry.intersection_update(photo)

        self._tags = tuple(sorted(date_tags))
        self._photometry = tuple(sorted(photometry))

    def __repr__(self):
        """Return repr of Date."""
        return "Date(mouse='{}', date={}, tags={}, photometry={})".format(
            self.mouse, self.date, self.tags, self.photometry)

    def __hash__(self):
        """Hash of a Date."""
        return hash(self.__repr__())

    def __str__(self):
        """Return str of Date."""
        return "{}_{}".format(self.mouse, self.date)

    def __lt__(self, other):
        """Less than."""
        if not isinstance(other, type(self)):
            raise NotImplementedError
        return self.mouse < other.mouse or \
            (self.mouse == other.mouse and self.date < other.date)

    def __le__(self, other):
        """Less than or equal to."""
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other):
        """Greater than."""
        return not self.__le__(other)

    def __ge__(self, other):
        """Greater than or equal to."""
        return not self.__lt__(other)

    def __eq__(self, other):
        """Equal."""
        return isinstance(other, type(self)) and self.mouse == other.mouse \
            and self.date == other.date

    def __ne__(self, other):
        """Not equal."""
        return not self.__eq__(other)

    def runs(self, run_types=None, runs=None, tags=None,
             exclude_tags='bad', name=None):
        """Return a RunSorter of associated runs.

        Can optionally filter runs by runtype or other tags.

        Parameters
        ----------
        run_types : list of str, optional
            List of run_types to include. Can also be a single type.
        runs : list of int
            List of run indices to include. Can also be a single index.
        tags : list of str, optional
            List of tags to filter on. Can also be a single tag.
        exclude_tags : list of str, optional
            List of tags to exclude. Can also be a single tag.
        name : str, optional
            Name of resulting RunSorter.

        Returns
        -------
        RunSorter

        """
        if name is None:
            name = str(self) + ' runs'

        if self._runs is None:
            meta_all = metadata.meta(mice=[self.mouse], dates=[self.date])
            meta_all_runs = meta_all.index.get_level_values('run')
            self._runs = {run: Run(mouse=self.mouse, date=self.date, run=run,
                                   cells=self.cells)
                          for run in meta_all_runs}
        else:
            for run in self._runs:
                self._runs[run].set_subset(self._cells)

        meta = metadata.meta(
            mice=[self.mouse], dates=[self.date], runs=runs,
            run_types=run_types, tags=tags, exclude_tags=exclude_tags)
        meta_runs = meta.index.get_level_values('run')

        run_objs = (self._runs[run] for run in meta_runs)

        return DateRunSorter(run_objs, name=name)

    def glm(self, glm_type='simpglm'):
        """Return GLM object.

        Returns
        -------
        GLM

        """
        if glm_type not in self._glm:
            self._glm[glm_type] = glm.glm(
                self.mouse, self.date, self.framerate, glm_type=glm_type)

            if self._cells is not None:
                self._glm[glm_type].subset(self._cells)

        return self._glm[glm_type]

    def clearcache(self):
        """Clear all cached data for this Date and any child Run objects."""
        if self._runs is not None:
            for run in self._runs.values():
                run.clearcache()
            self._runs = None
        self._glm = None


class Run(object):
    """A single run.

    Parameters
    ----------
    mouse : str
    date : int
    run : int
    cells : numpy array

    Attributes
    ----------
    mouse : str
    date : int
    run : int
    cells : numpy array
        A vector of cell numbers, used to reorder trace2p if comparing across
        days
    run_type : str
    tags : tuple of str
    parent : Date
        The parent Date object.

    Methods
    -------
    trace2p()
        Return the trace2p data for this Run.

    classify2p(newpars=None, randomize='')
        Return the classifier for this Run.

    """

    def __init__(self, mouse, date, run, cells=None):
        """Init."""
        self._mouse = str(mouse)
        self._date = int(date)
        self._run = int(run)
        self._cells = cells

        self._parent = Date(mouse=self.mouse, date=self.date)
        self._run_type, self._tags = None, None
        self._t2p, self._c2p, self._glm = None, None, None
        self._default_pars = None

    @property
    def mouse(self):
        """Name of mouse as string."""
        return copy(self._mouse)

    @property
    def date(self):
        """Date of Run as integer."""
        return copy(self._date)

    @property
    def run(self):
        """Index of run as integer."""
        return copy(self._run)

    @property
    def parent(self):
        """Link back to the parent Date."""
        return self._parent

    @property
    def run_type(self):
        """Run type of the Run as string."""
        if self._run_type is None:
            self._get_metadata()
        return copy(self._run_type)

    @property
    def tags(self):
        """Tuple of Run tags."""
        if self._tags is None:
            self._get_metadata()
        return tuple(self._tags)

    @property
    def cells(self):
        return copy(self._cells)

    def todict(self):
        """Return Run object as a dictionary representation."""
        return {'mouse': self.mouse, 'date': self.date, 'run': self.run}

    def set_subset(self, val=None):
        """
        Set the cell indices to be subset.

        Parameters
        ----------
        val : None or array
            Indices of cells to be kept, or all cells if val is None

        """

        self._cells = val

    def _get_metadata(self):
        """Query the metadata and set necessary properties."""
        meta = metadata.meta(
            mice=[self.mouse], dates=[self.date], runs=[self.run])
        assert len(meta) == 1
        tags = meta['tags'].iloc[0]
        run_type = meta['run_type'].iloc[0]
        self._run_type, self._tags = str(run_type), tuple(tags)

    def __repr__(self):
        """Return repr of Run."""
        return "Run(mouse='{}', date={}, run={}, run_type='{}', tags={})".format(
            self.mouse, self.date, self.run, self.run_type, self.tags)

    def __hash__(self):
        """Hash of a Run."""
        return hash(self.__repr__())

    def __str__(self):
        """Return str of Run."""
        return "{}_{}_{}".format(self.mouse, self.date, self.run)

    def __lt__(self, other):
        """Less than."""
        if not isinstance(other, type(self)):
            raise NotImplemented
        return self.mouse < other.mouse or \
            (self.mouse == other.mouse and self.date < other.date) or \
            (self.mouse == other.mouse and self.date == other.date and
             self.run < other.run)

    def __le__(self, other):
        """Less than or equal to."""
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other):
        """Greater than."""
        return not self.__le__(other)

    def __ge__(self, other):
        """Greater than or equal to."""
        return not self.__lt__(other)

    def __eq__(self, other):
        """Equal."""
        return isinstance(other, type(self)) and self.mouse == other.mouse \
            and self.date == other.date and self.run == other.run

    def __ne__(self, other):
        """Not equal."""
        return not self.__eq__(other)

    def trace2p(self):
        """Return trace2p data.

        Returns
        -------
        Trace2P

        """
        if self._t2p is None:
            self._t2p = paths.gett2p(
                self.mouse, self.date, self.run)

        self._t2p.subset(self._cells)

        return self._t2p

    def classify2p(self, newpars=None):
        """
        Return classifier.

        Parameters
        ----------
        newpars : dict
            Replace default parameters with values from this dict.

        Returns
        -------
        Classify2P

        """
        if newpars is None:
            if self._c2p is None:
                pars = self._default_classifier_pars()
                c2p_path = paths.getc2p(self.mouse, self.date, self.run, pars)
                self._c2p = classify2p.Classify2P(c2p_path, self, pars)
            return self._c2p
        else:
            pars = self._default_classifier_pars()
            pars.update(newpars)

            c2p_path = paths.getc2p(self.mouse, self.date, self.run, pars)
            return classify2p.Classify2P(c2p_path, self, pars)

    def _default_classifier_pars(self):
        """Return the default classifier parameters for the this Run."""
        if self._default_pars is None:
            pars = config.default()
            running_runs = self.parent.runs(run_types='running')
            training_runs = self.parent.runs(run_types='training')

            if self.run_type == 'training':
                # Could add specific parameters like this, but for now they
                # are the same.
                # pars.update({'analog-comparison-multiplier': 2.0,
                #              'remove-stim': True})
                pass

            pars.update({'mouse': self.mouse,
                         'comparison-date': str(self.date),
                         'comparison-run': self.run,
                         'training-date': str(self.date),
                         'training-other-running-runs':
                             [run.run for run in running_runs],
                         'training-runs': [run.run for run in training_runs]
                         })
            self._default_pars = pars
        return copy(self._default_pars)

    def clearcache(self):
        """Clear all cached data for this Run."""
        self._t2p, self._c2p, self._glm = None, None, None
        self._default_pars = None


class MouseSorter(UserList):
    """Iterator of Mouse objects.

    Parameters
    ----------
    mice : list of Mouse
        A list of Mouse objects to include.
    name : str, optional

    Attributes
    ----------
    name : str
        Name of MouseSorter

    Methods
    -------
    frommeta(mice=None, tags=None, name=None)
        Constructor to create a MouseSorter from metadata parameters.

    Notes
    -----
    Mice are sorted upon initialization so that iterating will always be
    sorted.

    """

    def __init__(self, mice=None, name=None):
        """Init."""
        if mice is None:
            mice = []
        self.data = sorted(mice)

        if name is None:
            self._name = None
        else:
            self._name = str(name)

    def __repr__(self):
        return "MouseSorter([{} {}], name={})".format(
            len(self), 'Mouse' if len(self) == 1 else 'Mice', self.name)

    @property
    def name(self):
        """The name of the MouseSorter, if it exists, else `None`."""
        if self._name is None:
            return 'None'
        else:
            return self._name

    @classmethod
    def frommeta(cls, mice=None, tags=None, exclude_tags='bad', name=None):
        """Initialize a MouseSorter from metadata.

        Parameters
        ----------
        mice : list of str, optional
            List of mice to include. Can also be a single mouse.
        tags : list of str, optional
            List of tags to filter on. Can also be a single tag.
        exclude_tags : list of str, optional
            List of tags to exclude. Can also be a single tag.
        name : str, optional
            A name to label the sorter, optional.


        Notes
        -----
        All arguments are used to filter the experimental metadata.
        All remaining mice will be included in the MouseSorter.

        Returns
        -------
        MouseSorter

        """
        meta = metadata.meta(mice=mice, tags=tags, exclude_tags=exclude_tags)
        meta_mice = meta.index.get_level_values('mouse').unique()

        mouse_objs = (Mouse(mouse=mouse) for mouse in meta_mice)

        return cls(date_objs, name=cls._create_name(mice, tags, name))

    @staticmethod
    def _create_name(mice, tags, name):
        """
        Create a name for saving unique to the inputs.

        Parameters
        ----------
        mice
        tags
        name

        Returns
        -------
        str

        """

        if name is not None:
            return name

        out = ''
        if tags is not None:
            if not isinstance(tags, list):
                tags = [tags]
            out += ','.join(tags)

        if mice is not None:
            if not isinstance(mice, list):
                mice = [mice]

            if len(out) > 0:
                out += '-'

            out += ','.join(mice)

        return out


class DateSorter(UserList):
    """Iterator of Date objects.

    Parameters
    ----------
    dates : list of Date
        A list of Date objects to include.
    name : str, optional

    Attributes
    ----------
    name : str
        Name of DateSorter

    Methods
    -------
    frommeta(mice=None, dates=None, tags=None, photometry=None, name=None)
        Constructor to create a DateSorter from metadata parameters.

    Notes
    -----
    Dates are sorted upon initialization so that iterating will always be
    sorted.

    """

    def __init__(self, dates=None, name=None):
        """Init."""
        if dates is None:
            dates = []
        self.data = sorted(dates)

        if name is None:
            self._name = None
        else:
            self._name = str(name)

    def __repr__(self):
        """Repr."""
        return "DateSorter([{} {}], name={})".format(
            len(self), 'Date' if len(self) == 1 else 'Dates', self.name)

    def __iter__(self):
        """Iter."""
        for date in self.data:
            yield date
            date.clearcache()

    @property
    def name(self):
        """The name of the DateSorter, if it exists, else `None`."""
        if self._name is None:
            return 'None'
        else:
            return self._name

    @classmethod
    def frommeta(
            cls, mice=None, dates=None, photometry=None, tags=None,
            exclude_tags='bad', name=None):
        """Initialize a DateSorter from metadata.

        Parameters
        ----------
        mice : list of str, optional
            List of mice to include. Can also be a single mouse.
        dates : list of int, optional
            List of dates to include. Can also be a single date.
        photometry : list of str, optional
            List of photometry labels to include. Can also be a single label.
        tags : list of str, optional
            List of tags to filter on. Can also be a single tag.
        exclude_tags : list of str, optional
            List of tags to exclude. Can also be a single tag.
        name : str, optional
            A name to label the sorter, optional.

        Notes
        -----
        All arguments are used to filter the experimental metadata.
        All remaining dates will be included in the DateSorter.

        Returns
        -------
        DateSorter

        """
        meta = metadata.meta(
            mice=mice, dates=dates, photometry=photometry, tags=tags,
            exclude_tags=exclude_tags)
        mouse_date_pairs = set(zip(meta.index.get_level_values('mouse'),
                                   meta.index.get_level_values('date')))

        date_objs = (Date(mouse=mouse, date=date)
                     for mouse, date in mouse_date_pairs)

        return cls(date_objs, name=cls._create_name(mice, dates, tags, name))

    @staticmethod
    def _create_name(mice, dates, tags, name):
        """
        Create a name for saving unique to the inputs.

        Parameters
        ----------
        mice
        dates
        tags
        name

        Returns
        -------
        str

        """

        if name is not None:
            return name

        out = ''
        if tags is not None:
            if not isinstance(tags, list):
                tags = [tags]
            out += ','.join(tags)

        if mice is not None:
            if not isinstance(mice, list):
                mice = [mice]

            if len(out) > 0:
                out += '-'

            out += ','.join(mice)

        if dates is not None:
            if not isinstance(dates, list):
                dates = [dates]

            if len(out) > 0:
                out += '-'

            out += ','.join([str(v) for v in dates])

        return out


class DatePairSorter(UserList):
    """Iterator of Date objects.

    Parameters
    ----------
    dates : list of Date
        A list of Date objects to include.
    name : str, optional

    Attributes
    ----------
    name : str
        Name of DatePairSorter

    Methods
    -------
    frommeta(mice=None, dates=None, tags=None, photometry=None, name=None)
        Constructor to create a DatePairSorter from metadata parameters.

    Notes
    -----
    Dates are sorted upon initialization so that iterating will always be
    sorted.

    """

    def __init__(self, dates=None, name=None):
        """Init."""
        if dates is None:
            dates = []
        self.data = sorted(dates)

        if name is None:
            self._name = None
        else:
            self._name = str(name)

    def __repr__(self):
        return "DatePairSorter([{} {}], name={})".format(
            len(self), 'Date pair' if len(self) == 1 else 'Date pairs',
            self.name)

    @property
    def name(self):
        """The name of the DatePairSorter, if it exists, else `None`."""
        if self._name is None:
            return 'None'
        else:
            return self._name

    @classmethod
    def frommeta(
            cls, mice=None, dates=None, photometry=None, tags=None,
            exclude_tags='bad', day_distance=None, sequential=True,
            cross_reversal=False, name=None):
        """Initialize a DatePairSorter from metadata.

        Parameters
        ----------
        mice : list of str, optional
            List of mice to include. Can also be a single mouse.
        dates : list of int, optional
            List of dates to include. Can also be a single date.
        tags : list of str, optional
            List of tags to filter on. Can also be a single tag.
        exclude_tags : list of str, optional
            List of tags to exclude. Can also be a single tag.
        photometry : list of str, optional
            List of photometry labels to include. Can also be a single label.
        day_distance : tuple of ints, optional
        sequential : bool, optional
        cross_reversal : bool
        name : str, optional
            A name to label the sorter, optional.

        Notes
        -----
        All arguments are used to filter the experimental metadata.
        All remaining dates will be included in the DatePairSorter.

        Returns
        -------
        DatePairSorter

        """
        meta = metadata.meta(
            mice=mice, dates=dates, photometry=photometry, tags=tags,
            exclude_tags=exclude_tags)
        meta['reversal'] = 0

        # Set reversal
        if not cross_reversal:
            for mouse in meta.index.get_level_values('mouse').unique():
                rev = metadata.reversal(mouse)
                meta.loc[Idx[mouse, rev:], 'reversal'] = 1

        # Iterate over pair-able dates
        pairs = []
        for (mouse, rev), ddf in meta.groupby(['mouse', 'reversal']):
            ds = np.array(ddf.index.get_level_values('date').unique())
            for d1 in ds:
                d2s = ds[ds > d1]
                if sequential and len(d2s) > 0:
                    d2s = [d2s[0]]

                for d2 in d2s:
                    tdelta = datetime.strptime(str(d2), '%y%m%d') \
                        - datetime.strptime(str(d1), '%y%m%d')
                    id1, id2 = xday.ids(mouse, d1, d2)
                    if len(id1) > 0 and \
                        (day_distance is None or
                         day_distance[0] <= tdelta.days <= day_distance[1]):
                        pairs.append((mouse, d1, d2, id1, id2))

        # Return a tuple of date tuples
        date_objs = ((Date(mouse=mouse, date=d1, cells=id1),
                      Date(mouse=mouse, date=d2, cells=id2))
                     for mouse, d1, d2, id1, id2 in pairs)

        return cls(date_objs, name=cls._create_name(mice, dates, tags, name))

    @staticmethod
    def _create_name(mice, dates, tags, name):
        """
        Create a name for saving unique to the inputs.

        Parameters
        ----------
        mice
        dates
        tags
        name

        Returns
        -------
        str

        """

        if name is not None:
            return name

        out = ''
        if tags is not None:
            if not isinstance(tags, list):
                tags = [tags]
            out += ','.join(tags)

        if mice is not None:
            if not isinstance(mice, list):
                mice = [mice]

            if len(out) > 0:
                out += '-'

            out += ','.join(mice)

        if dates is not None:
            if not isinstance(dates, list):
                dates = [dates]

            if len(out) > 0:
                out += '-'

            out += ','.join([str(v) for v in dates])

        return out


class RunSorter(UserList):
    """Iterator of Run objects.

    Parameters
    ----------
    runs : list of Run
        A list of Run objects to include.
    name : str, optional

    Notes
    -----
    Runs are sorted upon initialization so that iterating will always be
    sorted.

    """

    def __init__(self, runs=None, name=None):
        """Init."""
        if runs is None:
            runs = []
        self.data = sorted(runs)

        if name is None:
            self._name = None
        else:
            self._name = str(name)

    def __repr__(self):
        """Repr."""
        return "{}([{} {}], name={})".format(
            self.__class__.__name__, len(self),
            'Run' if len(self) == 1 else 'Runs', self.name)

    def __iter__(self):
        """Iter.

        Clears cached trace2p and classify2p from previous Run as you iterate.

        """
        for run in self.data:
            yield run
            run.clearcache()

    @property
    def name(self):
        """The name of the RunSorter, if it exists, else `None`."""
        if self._name is None:
            return 'None'
        else:
            return self._name

    def todicts(self):
        run_dicts = [run.todict() for run in self]
        return run_dicts

    def tojson(self, path):
        save_dict = {
            'name': self.name,
            'timestamp': timestamp(),
            'runs': self.todicts()}
        with open(path, 'w') as f:
            json.dump(save_dict, f, sort_keys=True, indent=2)

    @classmethod
    def fromdicts(cls, run_dicts, name=None):
        run_objs = [Run(**d) for d in run_dicts]
        return cls(run_objs, name=name)

    @classmethod
    def fromjson(cls, path, name=None):
        """Initialize a RunSorter from a saved JSON representation.
        """
        json_dict = json.load(path)
        if name is None:
            name = json_dict['name']
        return cls.fromdicts(json_dict['runs'], name=name)

    @classmethod
    def frommeta(
            cls, mice=None, dates=None, runs=None, run_types=None,
            photometry=None, tags=None, exclude_tags='bad', name=None):
        """Initialize a RunSorter from metadata.

        Parameters
        ----------
        mice : list of str, optional
            List of mice to include. Can also be a single mouse.
        dates : list of int, optional
            List of dates to include. Can also be a single date.
        runs : list of int, optional
            List of run indices to include. Can also be a single index.
        run_types : list of str, optional
            List of run_types to include. Can also be a single type.
        photometry : list of str, optional
            List of photometry labels to include. Can also be a single label.
        tags : list of str, optional
            List of tags to filter on. Can also be a single tag.
        exclude_tags : list of str, optional
            List of tags to exclude. Can also be a single tag.
        name : str, optional
            A name to label the sorter, optional.

        Notes
        -----
        All arguments are used to filter the experimental metadata.
        All remaining runs will be included in RunSorter.

        Returns
        -------
        RunSorter

        """
        meta = metadata.meta(
            mice=mice, dates=dates, runs=runs, run_types=run_types, tags=tags,
            exclude_tags=exclude_tags, photometry=photometry)
        mouse_date_run_pairs = set(zip(meta.index.get_level_values('mouse'),
                                       meta.index.get_level_values('date'),
                                       meta.index.get_level_values('run')))

        run_objs = (Run(mouse=mouse, date=date, run=run)
                    for mouse, date, run in mouse_date_run_pairs)

        return cls(run_objs, name=cls._create_name(mice, dates, runs, tags, name))

    @staticmethod
    def _create_name(mice, dates, runs, tags, name):
        """
        Create a name for saving unique to the inputs.

        Parameters
        ----------
        mice
        dates
        tags
        name

        Returns
        -------
        str

        """

        if name is not None:
            return name

        out = ''
        if tags is not None:
            if not isinstance(tags, list):
                tags = [tags]
            out += ','.join(tags)

        if mice is not None:
            if not isinstance(mice, list):
                mice = [mice]

            if len(out) > 0:
                out += '-'

            out += ','.join(mice)

        if dates is not None:
            if not isinstance(dates, list):
                dates = [dates]

            if len(out) > 0:
                out += '-'

            out += ','.join([str(v) for v in dates])

        if runs is not None:
            if not isinstance(runs, list):
                runs = [runs]

            if len(out) > 0:
                out += '-'

            out += ','.join([str(v) for v in runs])

        return out

    def dates(self, name=None):
        """
        Return DateSorter of all the parent Date objects.

        Returns
        -------
        DateSorter

        """
        if name is None:
            name = self.name + ' Dates'

        dates = {run.parent for run in self}

        return DateSorter(dates, name=name)


class DateRunSorter(RunSorter):
    """
    A RunSorter that does not clear cache after each iteration.

    Cache clearing is handled instead by a the DateSorter, allowing for data
    to remain cached through multiple iterations of child Run objects.

    """

    def __iter__(self):
        """Iter."""
        for run in self.data:
            yield run


def parse_name(args, cs=False):
    """Return a name based on the command line arguments passed.

    Specifically cares about mouse and date
    :param cs: if cs, include the cs values if passed in command line
    :return:

    """
    out = ''
    if 'mouse' in args:
        if isinstance(args.mouse, str):
            out += args.mouse
        else:
            out += ','.join(sorted(args.mouse))
    else:
        out += 'allmice'

    if 'group' in args:
        out += '-%s'%str(args.group)

    if 'training_date' in args:
        out += '-%s'%str(args.training_date)
    elif 'date' in args:
        out += '-%s'%str(args.date)

    if 'comparison_run' in args:
        if isinstance(args.comparison_run, int):
            out += '-%i'%args.comparison_run
        else:
            out += '-' + ','.join([str(i) for i in args.comparison_run])

    if cs and 'cs' in args:
        out += '-'
        if 'plus' in args.cs:
            out += 'p'
        if 'neutral' in args.cs:
            out += 'n'
        if 'minus' in args.cs:
            out += 'm'

    return out
