from builtins import object

from .. import paths, metadata

class Date(object):
    """A Day."""
    def __init__(self, mouse, date):
        self.mouse = mouse
        self.date = date

    def __repr__(self):
        """Return repr of Day."""
        return "Day(mouse={}, date={}, n_runs={})".format(
            self.mouse, self.date, len(self.runs))

    def __lt__(self, other):
        """Make dates sortable."""
        assert isinstance(other, type(self))
        return self.mouse <= other.mouse and self.date < other.date

    def __eq__(self, other):
        """Test equivalence."""
        return isinstance(other, type(self)) and self.mouse == other.mouse \
            and self.date == other.date

    @classmethod
    def allruns(
            cls, mouse, date, groups=None, running=True, training=True,
            spontaneous=True):
        """Return new Day object that automatically includes all runs.

        Parameters
        ----------
        mouse : str
            Mouse to include.
        date : str
            Date to include.
        groups : list of str
            Limit runs to a particular group.
        running : bool
            If True, include running runs.
        training : bool
            If True, include training runs.
        spontaneous : bool
            If True, include spontaneous runs.

        """
        runtypes = []
        if training:
            runtypes.append('train')
        if spontaneous:
            runtypes.append('spontaneous')
        if running:
            runtypes.append('running')

        meta = metadata.dataframe(
            mice=[mouse], dates=[date], groups=groups, runtypes=runtypes)

        run_nums = sorted(meta.run)

        runs = [Run(mouse=mouse, date=date, run=run) for run in run_nums]

        return cls(mouse=mouse, date=date, runs=runs)


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
