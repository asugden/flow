from builtins import object
from future.moves.collections import UserList

from .. import paths, metadata


class Date(object):
    """A Day."""
    def __init__(self, mouse, date):
        self.mouse = str(mouse)
        self.date = int(date)

    def __repr__(self):
        """Return repr of Day."""
        return "Day(mouse={}, date={})".format(self.mouse, self.date)

    def __lt__(self, other):
        """Make dates sortable."""
        assert isinstance(other, type(self))
        return self.mouse <= other.mouse and self.date < other.date

    def __eq__(self, other):
        """Test equivalence."""
        return isinstance(other, type(self)) and self.mouse == other.mouse \
            and self.date == other.date

    def runs(self, runtypes=None, tags=None):
        """Return a RunSorter of associated runs.

        Can optionally filter runs by runtype or other tags.

        Parameters
        ----------
        runtypes : list of {'train', 'spontaneous', 'running'}, optional
            List of runtypes to include. Defaults to all types.
        tags : list of str, optional
            List of tags to filter on.

        Returns
        -------
        RunSorter

        """
        meta = metadata.dataframe(
            mice=[self.mouse], dates=[self.date], tags=tags, runtypes=runtypes)

        run_nums = sorted(meta.run)

        runs = [Run(mouse=self.mouse, date=self.date, run=run)
                for run in run_nums]

        return RunSorter(runs)


class Run(object):
    """A run."""

    def __init__(self, mouse, date, run):
        self.mouse = str(mouse)
        self.date = int(date)
        self.run = int(run)

        self._t2p, self._classifier = None, None

    def __repr__(self):
        """Return repr of Run."""
        return 'Run(mouse={}, date={}, run={})'.format(
            self.mouse, self.date, self.run)

    def __lt__(self, other):
        """Make runs sortable."""
        assert isinstance(other, type(self))
        return self.mouse <= other.mouse and self.date <= other.date \
            and self.run < other.run

    def __eq__(self, other):
        """Test equivalence."""
        return isinstance(other, type(self)) and self.mouse == other.mouse \
            and self.date == other.date and self.run == other.run

    @property
    def t2p(self):
        if self._t2p is None:
            self._t2p = paths.gett2p(
                self.mouse, self.date, self.run)
        return self._t2p

    def classifier(self):
        pass


class DateSorter(UserList):
    def __init__(self, dates=None):
        if dates is None:
            dates = []
        self.data = sorted(dates)

    @classmethod
    def frommice(cls, mice):
        """Initialize a new RunSorter from a list of mice.

        Parameters
        ----------
        mice : list of str
            List of mice to include.

        """
        meta = metadata.dataframe(mice=mice)

        return cls(
            Date(m, d) for _, (m, d) in meta[['mouse', 'date']].groupby(
                ['mouse', 'date'], as_index=False).first().iterrows())


class RunSorter(UserList):
    def __init__(self, runs=None):
        if runs is None:
            runs = []
        self.data = sorted(runs)

    def __repr__(self):
        return "RunSorter({})".format(repr(self.data))

    def __str__(self):
        return "RunSorter with {} runs.".format(len(self.data))

    @classmethod
    def fromargs(cls, args):
        pass

    @classmethod
    def frommdrs(cls, mdrs):
        data = [Run(mouse, date, run) for mouse, date, run in mdrs]
        return cls(data)

    @classmethod
    def frommice(
            cls, mice, groups=None, training=False, spontaneous=True,
            running=False):
        """Initialize a new RunSorter from a list of mice.

        Parameters
        ----------
        mice : list of str
            List of mice to include.
        group : list of str
            If not None, limit to these groups.
        training : bool
            If True, include training trials.
        spontaneous : bool
            If True, include spontaneous trials.
        running : bool
            If True, include running trials.

        """
        if not training and not spontaneous and not running:
            raise ValueError('Must select at least 1 run type.')

        runtypes = []
        if training:
            runtypes.append('train')
        if spontaneous:
            runtypes.append('spontaneous')
        if running:
            runtypes.append('running')

        meta = metadata.dataframe(mice=mice, groups=groups, runtypes=runtypes)

        return cls(
            Run(m, d, r) for _, (m, d, r)
            in meta[['mouse', 'date', 'run']].iterrows())
