from builtins import object
from copy import copy
from datetime import datetime
from future.moves.collections import UserList
import numpy as np

from . import metadata
from .. import config, glm, paths, xday


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
        self._mouse = str(mouse)
        self._tags = None


    @property
    def mouse(self):
        return copy(self._mouse)


    @property
    def tags(self):
        if self._tags is None:
            self._get_metadata()
        return copy(self._tags)

    def _get_metadata(self):
        """Query the metadata and set necessary properties."""
        meta = metadata.meta(mice=[self.mouse])
        mouse_tags = set(meta['tags'].iloc[0])
        for tags in meta['tags']:
            mouse_tags.intersection_update(tags)

        self._tags = tuple(sorted(mouse_tags))

    def __repr__(self):
        """Return repr of Mouse."""
        return "Mouse(mouse={}, tags={})".format(self.mouse, self.tags)

    def __str__(self):
        """Return str of Mouse."""
        return self.mouse

    def __lt__(self, other):
        """Make mice sortable."""
        assert isinstance(other, type(self))
        return self.mouse < other.mouse

    def __eq__(self, other):
        """Test equivalence."""
        return isinstance(other, type(self)) and self.mouse == other.mouse

    def dates(self, dates=None, tags=None, name=None):
        """Return a DateSorter of associated Dates.

        Can optionally filter Dates by tags.

        Parameters
        ----------
        dates : list of int, optional
            List of dates to filter on.
        tags : list of str, optional
            List of tags to filter on.
        name : str, optional
            Name of resulting DateSorter.

        Returns
        -------
        DateSorter

        """
        meta = metadata.meta(mice=[self.mouse], dates=dates, tags=tags)

        date_objs = (Date(mouse=self.mouse, date=date)
                     for date in meta['date'].unique())

        return DateSorter(date_objs, name=name)


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
        A vector of cell numbers, used to reorder trace2p if comparing across days
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
        self._mouse = str(mouse)
        self._date = int(date)
        self._cells = cells

        self._parent = Mouse(mouse=self.mouse)
        self._tags, self._photometry = None, None
        self._runs = None

    @property
    def mouse(self):
        return copy(self._mouse)

    @property
    def date(self):
        return copy(self._date)

    @property
    def parent(self):
        return self._parent

    @property
    def tags(self):
        if self._tags is None:
            self._get_metadata()
        return copy(self._tags)

    @property
    def cells(self):
        return copy(self._cells)

    @property
    def photometry(self):
        if self._photometry is None:
            self._get_metadata()
        return copy(self._photometry)

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
        return "Date(mouse={}, date={}, tags={}, photometry={})".format(
            self.mouse, self.date, self.tags, self.photometry)

    def __str__(self):
        """Return str of Date."""
        return "{}_{}".format(self.mouse, self.date)

    def __lt__(self, other):
        """Make dates sortable."""
        assert isinstance(other, type(self))
        return self.mouse < other.mouse or \
               (self.mouse == other.mouse and self.date < other.date)

    def __eq__(self, other):
        """Test equivalence."""
        return isinstance(other, type(self)) and self.mouse == other.mouse \
            and self.date == other.date

    def runs(self, run_types=None, runs=None, tags=None, name=None):
        """Return a RunSorter of associated runs.

        Can optionally filter runs by runtype or other tags.

        Parameters
        ----------
        run_types : list of {'training', 'spontaneous', 'running'}, optional
            List of run_types to include. Defaults to all types. Can also be
            a single run_type.
        runs : list of int
            List of run numbers to include. Defaults to all runs.
        tags : list of str, optional
            List of tags to filter on.
        name : str, optional
            Name of resulting RunSorter.

        Returns
        -------
        RunSorter

        """
        if run_types is not None and not isinstance(run_types, list):
            run_types = [run_types]

        if self._runs is None:
            meta = metadata.meta(mice=[self.mouse], dates=[self.date])
            self._runs = {run: Run(mouse=self.mouse, date=self.date, run=run, cells=self.cells)
                          for run in meta['run']}

        meta = metadata.meta(
            mice=[self.mouse], dates=[self.date], runs=runs, tags=tags,
            run_types=run_types)

        run_objs = (self._runs[run] for run in meta['run'])

        return RunSorter(run_objs, name=name)

    def glm(self):
        """Return GLM object.

        Returns
        -------
        GLM

        """
        if self._glm is None:
            self._glm = glm.glm(self.mouse, self.date)

            if self._cells is not None:
                self._glm.subset(self._cells)

        return self._glm


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
        A vector of cell numbers, used to reorder trace2p if comparing across days
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
        self._mouse = str(mouse)
        self._date = int(date)
        self._run = int(run)
        self._cells = cells

        self._parent = Date(mouse=self.mouse, date=self.date)
        self._run_type, self._tags = None, None
        self._t2p, self._c2p, self._glm = None, None, None

    @property
    def mouse(self):
        return copy(self._mouse)

    @property
    def date(self):
        return copy(self._date)

    @property
    def run(self):
        return copy(self._run)

    @property
    def parent(self):
        return self._parent

    @property
    def run_type(self):
        if self._run_type is None:
            self._get_metadata()
        return copy(self._run_type)

    @property
    def tags(self):
        if self._tags is None:
            self._get_metadata()
        return copy(self._tags)

    @property
    def cells(self):
        return copy(self._cells)

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
        return 'Run(mouse={}, date={}, run={}, run_type={}, tags={})'.format(
            self.mouse, self.date, self.run, self.run_type, self.tags)

    def __str__(self):
        """Return str of Run."""
        return "{}_{}_{}".format(self.mouse, self.date, self.run)

    def __lt__(self, other):
        """Make runs sortable, by (mouse, date, run)."""
        assert isinstance(other, type(self))
        return self.mouse < other.mouse or \
               (self.mouse == other.mouse and self.date < other.date) or \
               (self.mouse == other.mouse and self.date == other.date and
                self.run < other.run)

    def __eq__(self, other):
        """Test equivalence."""
        return isinstance(other, type(self)) and self.mouse == other.mouse \
            and self.date == other.date and self.run == other.run

    def trace2p(self):
        """Return trace2p data.

        Returns
        -------
        Trace2P

        """
        if self._t2p is None:
            self._t2p = paths.gett2p(
                self.mouse, self.date, self.run)

            if self._cells is not None:
                self._t2p.subset(self._cells)

        return self._t2p

    def classify2p(self, newpars=None, randomize=''):
        """Return classifier.

        Parameters
        ----------
        newpars : dict
            Replace default parameters with values from this dict.
        randomize

        Returns
        -------
        Classify2P

        """
        pars = config.default()
        running_runs = metadata.meta(
            mice=[self.mouse], dates=[self.date], run_types=['running'])
        training_runs = metadata.meta(
            mice=[self.mouse], dates=[self.date], run_types=['training'])
        # TODO: Add option to train on a different day
        pars.update({'mouse': self.mouse,
                     'comparison-date': str(self.date),
                     'comparison-run': self.run,
                     'training-date': str(self.date),
                     'training-other-running-runs': sorted(running_runs.run),
                     'training-runs': sorted(training_runs.run)})

        if newpars is None:
            if self._c2p is None:
                self._c2p = paths.classifier2p(
                    self.mouse, self.date, self.run, pars, randomize)
            return self._c2p
        else:
            for key in newpars:
                pars[key] = newpars[key]

            return paths.classifier2p(
                    self.mouse, self.date, self.run, pars, randomize)


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
    Mice are sorted upon initialization so that iterating will always be sorted.

    """
    def __init__(self, mice=None, name=None):
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
        """The name of the MouseSorter, if it exists, else `None`"""
        if self._name is None:
            return 'None'
        else:
            return self._name

    @classmethod
    def frommeta(cls, mice=None, tags=None, name=None):
        """Initialize a MouseSorter from metadata.

        Parameters
        ----------
        mice : list of str, optional
        tags : list of str, optional
        name : str, optional
            Name/label for Sorter.

        Notes
        -----
        All arguments are used to filter the experimental metadata.
        All remaining mice will be included in the MouseSorter.

        Returns
        -------
        MouseSorter

        """
        meta = metadata.meta(mice=mice, tags=tags)

        mouse_objs = (Mouse(mouse=mouse) for mouse in meta.mouse.unique())

        return cls(mouse_objs, name=name)


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
    Dates are sorted upon initialization so that iterating will always be sorted.

    """
    def __init__(self, dates=None, name=None):
        if dates is None:
            dates = []
        self.data = sorted(dates)

        if name is None:
            self._name = None
        else:
            self._name = str(name)

    def __repr__(self):
        return "DateSorter([{} {}], name={})".format(
            len(self), 'Date' if len(self) == 1 else 'Dates', self.name)

    @property
    def name(self):
        """The name of the DateSorter, if it exists, else `None`"""
        if self._name is None:
            return 'None'
        else:
            return self._name

    @classmethod
    def frommeta(
            cls, mice=None, dates=None, tags=None, photometry=None, name=None):
        """Initialize a DateSorter from metadata.

        Parameters
        ----------
        mice : list of str, optional
        dates : list of int, optional
        tags : list of str, optional
        photometry : list of str, optional
        name : str, optional
            Name/label for Sorter.

        Notes
        -----
        All arguments are used to filter the experimental metadata.
        All remaining dates will be included in the DateSorter.

        Returns
        -------
        DateSorter

        """
        meta = metadata.meta(
            mice=mice, dates=dates, tags=tags, photometry=photometry)

        date_objs = (
            Date(mouse=date_df.mouse, date=date_df.date) for _, date_df in
            meta.groupby(['mouse', 'date'], as_index=False).first().iterrows())

        return cls(date_objs, name=name)


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
    Dates are sorted upon initialization so that iterating will always be sorted.

    """

    def __init__(self, dates=None, name=None):
        if dates is None:
            dates = []
        self.data = sorted(dates)

        if name is None:
            self._name = None
        else:
            self._name = str(name)

    def __repr__(self):
        return "DatePairSorter([{} {}], name={})".format(
            len(self), 'Date' if len(self) == 1 else 'Dates', self.name)

    @property
    def name(self):
        """The name of the DatePairSorter, if it exists, else `None`"""
        if self._name is None:
            return 'None'
        else:
            return self._name

    @classmethod
    def frommeta(
            cls, mice=None, dates=None, tags=None, photometry=None,
            day_distance=None, sequential=True, cross_reversal=False, name=None):
        """Initialize a DatePairSorter from metadata.

        Parameters
        ----------
        mice : list of str, optional
        dates : list of int, optional
        tags : list of str, optional
        photometry : list of str, optional
        day_distance : tuple of ints, optional
        sequential : bool, optional
        cross_reversal : bool
        name : str, optional
            Name/label for Sorter.

        Notes
        -----
        All arguments are used to filter the experimental metadata.
        All remaining dates will be included in the DatePairSorter.

        Returns
        -------
        DatePairSorter

        """
        meta = metadata.meta(
            mice=mice, dates=dates, tags=tags, photometry=photometry)
        meta['reversal'] = 0

        # Set reversal
        if not cross_reversal:
            for mouse in [ddf.mouse for _, ddf in
                          meta.groupby('mouse', as_index=False).first().iterrows()]:
                rev = metadata.reversal(mouse)
                meta.loc[(meta['mouse'] == mouse) & (meta['date'] >= rev), 'reversal'] = 1

        # Iterate over pair-able dates
        pairs = []
        for mouse, rev in [(ddf.mouse, ddf.reversal) for _, ddf in
                           meta.groupby(['mouse', 'reversal'], as_index=False).first().iterrows()]:
            ds = np.array([ddf.date for _, ddf in meta.loc[(meta['mouse'] == mouse)
                          & (meta['reversal'] == rev), :].groupby('date', as_index=False).first().iterrows()])

            for d1 in ds:
                d2s = ds[ds > d1]
                if sequential and len(d2s) > 0:
                    d2s = [d2s[0]]

                for d2 in d2s:
                    tdelta = datetime.strptime(str(d2), '%y%m%d') \
                             - datetime.strptime(str(d1), '%y%m%d')
                    id1, id2 = xday.ids(mouse, d1, d2)
                    if day_distance[0] <= tdelta.days <= day_distance[1] and len(id1) > 0:
                        pairs.append((mouse, d1, d2, id1, id2))

        # Return a tuple of date tuples
        date_objs = ((Date(mouse=mouse, date=d1, cells=id1), Date(mouse=mouse, date=d2, cells=id2))
                     for mouse, d1, d2, id1, id2 in pairs)

        return cls(date_objs, name=name)


class RunSorter(UserList):
    """Iterator of Run objects.

    Parameters
    ----------
    runs : list of Run
        A list of Run objects to include.
    name : str, optional

    Attributes
    ----------
    name : str
        Name of DateSorter

    Methods
    -------
    frommeta(mice=None, dates=None, runs=None, run_types=None, tags=None,
             photometry=None, name=None)
        Constructor to create a RunSorter from metadata parameters.

    Notes
    -----
    Runs are sorted upon initialization so that iterating will always be sorted.

    """
    def __init__(self, runs=None, name=None):
        if runs is None:
            runs = []
        self.data = sorted(runs)

        if name is None:
            self._name = None
        else:
            self._name = str(name)

    def __repr__(self):
        return "RunSorter([{} {}], name={})".format(
            len(self), 'Run' if len(self) == 1 else 'Runs', self.name)

    @property
    def name(self):
        """The name of the RunSorter, if it exists, else `None`"""
        if self._name is None:
            return 'None'
        else:
            return self._name

    # @classmethod
    # def fromargs(cls, args):
    #     """Initialize a RunSorter """
    #     name = parse_name(args)
    #
    #     # TODO: how to handle run type?
    #
    #     return cls.frommeta(mice=args.mice, dates=args.dates, name=name)


    @classmethod
    def frommeta(
            cls, mice=None, dates=None, runs=None, run_types=None, tags=None,
            photometry=None, name=None):
        """Initialize a RunSorter from metadata.

        Parameters
        ----------
        mice : list of str, optional
            List of mice to include.
        dates : list of int, optional
            List of dates to include.
        runs : list of int, optional
            List of run indices to include.
        run_types : list of str, optional
            List of run_types to include.
        tags : list of str, optional
            List of tags that must be present.
        photometry : list of str, optional
            List of photometry labels that must be present.
        name : str, optional
            A name to label the sorter, optional.

        Notes
        -----
        All arguments are used to filter the experiemntal metadata.
        All remaining runs will be included in RunSorter.

        Returns
        -------
        RunSorter

        """
        meta = metadata.meta(
            mice=mice, dates=dates, runs=runs, run_types=run_types, tags=tags,
            photometry=photometry)

        run_objs = (Run(mouse=run.mouse, date=run.date, run=run.run)
                    for _, run in meta.iterrows())

        return cls(run_objs, name=name)

    # @classmethod
    # def frommice(
    #         cls, mice=None, dates=None, training=False, spontaneous=False,
    #         running=False, name=None):
    #     """Initialize a new RunSorter from a list of mice.
    #
    #     Parameters
    #     ----------
    #     mice : list of str, optional
    #         List of mice to include.
    #     dates : list of int, optional
    #         If not None, limit to these dates.
    #     training : bool
    #         If True, include training trials.
    #     spontaneous : bool
    #         If True, include spontaneous trials.
    #     running : bool
    #         If True, include running trials.
    #     name : str, optional
    #         A name to label the sorter, optional.
    #
    #     """
    #     if not training and not spontaneous and not running:
    #         raise ValueError('Must select at least 1 run type.')
    #
    #     run_types = []
    #     if training:
    #         run_types.append('training')
    #     if spontaneous:
    #         run_types.append('spontaneous')
    #     if running:
    #         run_types.append('running')
    #
    #     return cls.frommeta(mice=mice, dates=dates, run_types=run_types)

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
