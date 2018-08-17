from future.moves.collections import UserList

from . import core
from .. import metadata


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
            core.Date(m, d) for _, (m, d) in meta[['mouse', 'date']].groupby(
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
        data = [core.Run(mouse, date, run) for mouse, date, run in mdrs]
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
            core.Run(m, d, r) for _, (m, d, r)
            in meta[['mouse', 'date', 'run']].iterrows())


if __name__ == '__main__':
    from ... import metadata
    mdrs = metadata.sortedall()[:5]
    z = RunSorter.frommdrgs(mdrs)

    from pudb import set_trace; set_trace()
