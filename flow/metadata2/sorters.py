from builtins import object
from future.moves.collections import UserList

from . import metadata
from .. import paths


class Date(object):
    """A Day."""
    def __init__(self, mouse, date, tags=None):
        self.mouse = str(mouse)
        self.date = int(date)
        if tags is None:
            self.tags = ()
        else:
            self.tags = tuple(tags)

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

    def runs(self, run_types=None, tags=None):
        """Return a RunSorter of associated runs.

        Can optionally filter runs by runtype or other tags.

        Parameters
        ----------
        run_types : list of {'training', 'spontaneous', 'running'}, optional
            List of run_types to include. Defaults to all types.
        tags : list of str, optional
            List of tags to filter on.

        Returns
        -------
        RunSorter

        """
        meta = metadata.meta(
            mice=[self.mouse], dates=[self.date], tags=tags,
            run_types=run_types, sort=True)

        runs = [Run(mouse=self.mouse, date=self.date, run=run, run_type=run_type)
                for _, (run, run_type) in meta[['run', 'run_type']].iterrows()]

        return RunSorter(runs)


class Run(object):
    """A run."""

    def __init__(self, mouse, date, run, run_type, tags=None):
        self.mouse = str(mouse)
        self.date = int(date)
        self.run = int(run)
        self.run_type = str(run_type)
        if tags is None:
            self.tags = ()
        else:
            self.tags = tuple(tags)

        self._t2p, self._c2p = None, None

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

    def trace2p(self):
        if self._t2p is None:
            self._t2p = paths.gett2p(
                self.mouse, self.date, self.run)
        return self._t2p

    def classify2p(self, newpars=None, randomize=''):
        pars = config.default()

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



class DateSorter(UserList):
    def __init__(self, dates=None, name=None):
        if dates is None:
            dates = []
        self.data = sorted(dates)

        if name is None:
            self._name = None
        else:
            self._name = str(name)

    def __repr__(self):
        return "DateSorter({}, name={})".format(repr(self.data), self.name)

    def __str__(self):
        return "DateSorter with {} dates.".format(len(self.data))

    @property
    def name(self):
        """The name of the DateSorter, if it exists, else `nameless`"""
        if self._name is None:
            return 'nameless'
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
        All arguments are used to filter the experiemntal metadata.
        All remaining dates will be included in DateSorter.

        Returns
        -------
        DateSorter

        """
        meta = metadata.meta(
            mice=mice, dates=dates, tags=tags, photometry=photometry, sort=True)


        date_objs = []
        for (mouse, date), date_df in meta.groupby(['mouse', 'date'], as_index=False):
            date_tags = set(date_df['tags'].iloc[0])
            for tags in date_df['tags']:
                date_tags.intersection_update(tags)
            date_objs.append(Date(mouse, date, tags=date_tags))

        return cls(date_objs, name=name)


class RunSorter(UserList):
    def __init__(self, runs=None, name=None):
        if runs is None:
            runs = []
        self.data = sorted(runs)

        if name is None:
            self._name = None
        else:
            self._name = str(name)

    def __repr__(self):
        return "RunSorter({}, name={})".format(repr(self.data), self.name)

    def __str__(self):
        return "RunSorter with {} runs.".format(len(self.data))

    @property
    def name(self):
        """The name of the RunSorter, if it exists, else `nameless`"""
        if self._name is None:
            return 'nameless'
        else:
            return self._name

    @classmethod
    def fromargs(cls, args):
        """Initialzie a RunSorter """
        name = parse_name(args)

        # TODO: how to handle run type?

        return cls.frommice(mice=args.mice, dates=args.dates, name=name)


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
            photometry=photometry, sort=True)

        return cls(
            (Run(mouse=run['mouse'], date=run['date'], run=run['run'],
                 run_type=run['run_type'], tags=run['tags']) for _, run
                    in meta.iterrows()),
            name=name)

    @classmethod
    def frommice(
            cls, mice=None, dates=None, training=False, spontaneous=True,
            running=False, name=None):
        """Initialize a new RunSorter from a list of mice.

        Parameters
        ----------
        mice : list of str, optional
            List of mice to include.
        dates : list of int, optional
            If not None, limit to these dates.
        training : bool
            If True, include training trials.
        spontaneous : bool
            If True, include spontaneous trials.
        running : bool
            If True, include running trials.
        name : str, optional
            A name to label the sorter, optional.

        """
        if not training and not spontaneous and not running:
            raise ValueError('Must select at least 1 run type.')

        run_types = []
        if training:
            run_types.append('training')
        if spontaneous:
            run_types.append('spontaneous')
        if running:
            run_types.append('running')

        return cls.frommeta(mice=mice, dates=dates, run_types=run_types)

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
