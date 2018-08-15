from __future__ import print_function
from six import with_metaclass

from abc import ABCMeta, abstractmethod
import os
from importlib import import_module
import numpy as np
from copy import deepcopy

from ... import hardcodedpaths
# from ... import parseargv
from ... import metadata


class BackendBase(with_metaclass(ABCMeta, object)):
    def __init__(self, **kwargs):
        """
        Organized in three ways:
        'day' First, a list of days and the resulting cell-ids
        'id-day-run' a list of cell-id-day-days for run specific analyses
        'id-day' Second, a list of cell-id-days for day-specific analyses
        'id' Third, a database of cell-ids and the resulting analyses
        'imported-days' Fourth, a list of imported days

        """
        self._loadanalyzers()
        self._initialize(**kwargs)

        self._indmouse = ''
        self._inddate = -1

    @abstractmethod
    def _initialize(self, **kwargs):
        """Initialization steps customizable by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def store(self, analysis_name, data, keys, dependents=None):
        """Store a value from running an analysis in the data store."""
        raise NotImplementedError

    def store_all(self, data_dict, keys, dependencies=None):
        """Store a set of key, value pairs at once.

        This can potentially be implemented more efficiently by individual
        database backends.

        """
        for key, val in data_dict.iteritems():
            self.store(key, val, keys, dependencies.get(key, {}))

    @abstractmethod
    def recall(self, analysis_name, **keys):
        """Return the value from the data store for a given analysis."""
        raise NotImplementedError

    @abstractmethod
    def is_analysis_old(self, analysis_name, keys):
        """Determine if the analysis needs to be re-run."""
        return True

    def save(self, **kwargs):
        """Save all updated databases.

        Required by some backends.

        """
        pass

    #
    # End of customizable functions.
    #

    def analysisname(self, analysis, cs):
        """Determine whether the analysis name is cs-specific.

        :param analysis: analysis name, str
        :param cs: stimulus, str
        :return: analysis name, corrected for cs, or None

        """
        if analysis in self.ans:
            return analysis
        elif '%s-%s' % (analysis, cs) in self.ans:
            return '%s-%s' % (analysis, cs)
        else:
            return None

    def md(self, mouse, date=-1):
        """
        Set the mouse and date for indexing as a dict
        :param mouse: mouse name, str or list of inputs
        :param date: date, str or int
        :return:
        """

        # The output of sortedruns and sorteddays is a metadata list, account for this
        if (isinstance(mouse, list) or isinstance(mouse, tuple))and len(mouse) > 1:
            date = mouse[1]
            mouse = mouse[0]

        # Format correctly in case people forget
        if isinstance(date, str):
            date = int(date)

        self._indmouse = mouse
        self._inddate = date

    def __getitem__(self, analysis):
        """
        Use dict-style indexing into the array, mouse and date must already be set via md.
        :param analysis: analysis name
        :return: output of self.get()
        """

        if len(self._indmouse) == 0 or self._inddate < 0:
            raise ValueError('Mouse and date not set via md()')

        return self.get(analysis, self._indmouse, self._inddate)

    def get(self, analysis, mouse, date, run=-1, force=False):
        """Get an analysis.

        :param analysis: string designating analysis to be returned
        :param mouse: mouse name, string
        :param date: mouse date, int
        :param run: mouse run, int (optional for analyses that do not require run)
        :param force: force reanalysis
        :param classifiers: alternative classifier(s) to be used if not None
        :return: analysis, if it exists

        """
        from ... import parseargv
        # Check if analysis exists
        if analysis not in self.ans:
            return None

        # # Format correctly
        # if date is None:
        #     date = mouse
        #     mouse = self.m
        # TODO: remove. should always pass in date as an integer
        if isinstance(date, str):
            date = int(date)
        # self.m = mouse

        self._indmouse, self._inddate = mouse, date

        # Get the keyword of the analysis
        an = self.ans[analysis]
        keys = {}
        keys['mouse'] = mouse

        keys['date'] = date
        if an['across'] == 'day':
            pass
        elif an['across'] == 'run':
            keys['run'] = run

        # Add the classifier name if possible, can use passed pars to generate name
        if 'classifier' in an['requires']:
            pars = parseargv.parsekv(['-mouse', mouse, '-date', '%i' % (date)])
            keys['classifier_word'] = hardcodedpaths.classifierword(pars)

        keys['updated'] = an['updated']

        if force or self.is_analysis_old(analysis, keys):
            md = metadata.md(mouse, date)
            mdr = metadata.mdr(mouse, date, md[0])
            mdr['spontaneous'], mdr['run'] = md, run
            mdr['hungry'], mdr['sated'] = metadata.hungrysated(mouse, date)
            if 'classifier' not in an['requires']:
                pars = parseargv.parsekv(
                    ['-mouse', mouse, '-date', '%i' % (date)])

            gdf = GetDateFiles(pars)
            ga = GetAnalysis(self, mouse, date, run)
            c = object.__new__(an['class'])

            print('\tupdating analysis...', c)

            # Inject methods
            setattr(c, 'trace2p', gdf.trace2p)
            setattr(c, 'classifier', gdf.classifier)
            setattr(c, 'pars', pars)
            setattr(c, 'analysis', ga.analyze)
            setattr(c, 'andb', self)
            c.__init__(mdr)

            out = c.get()
            # for key, value in out.iteritems():
            #     key_keys = check_dependencies(out, keys)
            #     self.store(key, value, key_keys)
            self.store_all(out, keys, self.deps)

        out = deepcopy(self.recall(analysis, keys))

        if isinstance(out, float) and np.isnan(out):
            return None
        if isinstance(out, list) and len(out) == 0:
            return None
        if isinstance(out, np.ndarray) and len(out) == 0:
            return None
        return out

    def update(self, mouse, dates=None, force=False, falsepositives=False):
        """Update and run all analyses for mouse.

        :param mouse:
        :return:

        """
        from ... import parseargv
        # if len(mouse) > 0:
        #     self.m = mouse
        # if self.m not in self.dbrs:
        #     self._open(self.m)
        if dates is None:
            dates = metadata.dates(mouse)
        if isinstance(dates, str) or isinstance(dates, int):
            dates = [dates]

        for date in dates:
            if isinstance(date, str):
                date = int(date)

            self._indmouse, self._inddate = mouse, date

            pars = parseargv.parsekv(['-mouse', mouse, '-date', '%i' % (date)])
            gdf = GetDateFiles(pars)
            ga = GetAnalysis(self, mouse, date, -1)

            for c in self.clans:
                if (c.__name__ != 'FalsePositives' and
                    c.__name__ != 'ReplayFP' and
                    c.__name__ != 'ReplayFPEvents' and
                    c.__name__ != 'FalsePositivesCorr') or \
                        falsepositives:

                    keys = {}
                    keys['mouse'] = mouse

                    keys['date'] = date
                    if getattr(c, 'across') == 'day':
                        pass
                    elif getattr(c, 'across') == 'run':
                        print('ERROR, have not implemented days.')
                        exit(0)

                    if 'classifier' in getattr(c, 'requires'):
                        keys['classifier_word'] = \
                            hardcodedpaths.classifierword(pars)
                    keys['updated'] = deepcopy(getattr(c, 'updated'))

                    fnd = True
                    for an in self._flatten(getattr(c, 'sets')):
                        if self.is_analysis_old(an, keys):
                            fnd = False
                            break
                        # if '%s-%s' % (key, an) not in self.dbus[mouse]:
                        #     fnd = False
                        # elif self.dbus[mouse]['%s-%s' % (key, an)] != andate:
                        #     fnd = False

                    if not fnd or force:
                        print('\tupdating analysis...', c)
                        md = metadata.md(mouse, date)
                        mdr = metadata.mdr(mouse, date, md[0])
                        mdr['spontaneous'], mdr['run'] = md, -1
                        mdr['hungry'], mdr['sated'] = \
                            metadata.hungrysated(mouse, date)

                        co = object.__new__(c)
                        # Inject methods
                        setattr(co, 'trace2p', gdf.trace2p)
                        setattr(co, 'classifier', gdf.classifier)
                        setattr(co, 'pars', pars)
                        setattr(co, 'analysis', ga.analyze)
                        setattr(c, 'andb', self)
                        co.__init__(mdr)

                        out = co.get()
                        self.store_all(out, keys, self.deps)
                        # for name in out:
                        #     self.dbrs[mouse]['%s-%s' % (key, name)] = out[name]
                        # for name in out:
                        #     self.dbus[mouse][key + '-%s' % (name)] = deepcopy(andate)
                        # for name in out:
                        #     self.updated_analyses[mouse].append('%s-%s' % (key, name))

    def _loadanalyzers(self):
        """Load analysis modules.

        :return:

        """
        self.ans, self.clans = self._linkanalyzers(self._getanalyzers())
        self.deps = self._determine_dependents()

    # Link all analyzer classes them with the appropriate protocols and what they set
    def _linkanalyzers(self, anclasses):
        out = {}
        for c in anclasses:
            for key in self._flatten(getattr(c, 'sets')):
                out[key] = {
                    'class': c,
                    'requires': getattr(c, 'requires'),
                    'across': getattr(c, 'across'),
                    'updated': getattr(c, 'updated', ''),
                    'depends_on': getattr(c, 'depends_on', tuple()),
                }
        return out, anclasses

    def _flatten(self, nestedlist):
        """
        Convert a possibly nested list into a flat list
        :param nestedlist: possibly nested list
        :return: flattened list
        """

        if not isinstance(nestedlist, list):
            return [nestedlist]
        else:
            out = []
            for el in nestedlist:
                out.extend(self._flatten(el))
            return out

    def _getanalyzers(self, test=False):
        """Load all analyzer classes from all modules from a path and
        return the classes.

        """
        out = []
        fpath = os.path.join(
            os.path.dirname(__file__), '..', '..', 'analyses')
        if os.path.exists(fpath):
            files = os.listdir(fpath)
            for f in files:
                if f[-3:] == '.py' and f[0] != '_':
                    module = import_module(
                        '.analyses.%s' % (f[:-3]), package='replay.lib')
                    out.extend(self._screenanalyzer(module))
        return out

    def _screenanalyzer(self, module):
        """
        Screen files for classes that have the 'requires' parameter
        :param module: a recently imported python module
        :return: the classes the module defines
        """
        classes = [getattr(module, m) for m in vars(module) if m[0] != '_' and
                   isinstance(getattr(module, m, None), type)]
        cutclasses = [c for c in classes if hasattr(c, 'requires')]
        return cutclasses


    def _determine_dependents(self):
        dependents = {}
        for an, an_dict in self.ans.iteritems():
            for dep_an in an_dict.get('depends_on', {}):
                if dep_an in dependents:
                    dependents[dep_an].append(an)
                else:
                    dependents[dep_an] = [an]
        return dependents


def keyname(analysis, keys):

        keyname = '%s-%i' % (keys['mouse'], int(keys['date']))

        if 'run' in keys:
            keyname += '%02i' % (keys['run'])

        if 'classifier_word' in keys:
            keyname += '-%s' % (keys['classifier_word'])

        keyname += '-%s' % (analysis)

        return keyname


class GetDateFiles():
    def __init__(self, pars, force=True):
        """
        A class that holds loaded trace2p and classifier files so that they can be reused between analyses. Loads only
        upon necessity.
        :param pars: parameters dict, from settings
        """

        self.t2ps = {}
        self.classifiers = {}
        self.pars = pars
        self.mouse = pars['mouse']
        self.date = pars['comparison-date']
        self.force = force

    def trace2p(self, run):
        """
        Get trace2p file
        :param run: run number, int
        :return: t2p instance
        """

        if run not in self.t2ps:
            self.t2ps[run] = hardcodedpaths.gett2p(self.mouse, self.date, run)
        return self.t2ps[run]

    def classifier(self, run, randomize='', n=1):
        """
        Get classifier output
        :param run: run number, int
        :return: classifier dict
        """
        from ... import parseargv
        if n <= 1 and len(randomize) == 0:
            if run not in self.classifiers:
                self.pars['comparison-run'] = run
                self.classifiers[run] = parseargv.classifier(
                    self.pars, randomize, force=self.force)
            return self.classifiers[run]
        elif n <= 1:
            return parseargv.classifier(self.pars, randomize, force=self.force)
        else:
            self.pars['comparison-run'] = run
            return parseargv.classifiers(self.pars, randomize, n)


class GetAnalysis():
    def __init__(self, andb, mouse, date, run=-1):
        """
        Make a class that holds the analysis database, mouse, date, and, run for calling within analysis functions
        :param andb: "self" from the analysis db
        :param mouse: name of the mouse
        :param date: date, str
        :param run: run, int, set to -1 if ignored
        """

        self.db = andb
        self.mouse = mouse
        self.date = date
        self.run = run

    def analyze(self, name):
        """
        Return an analysis to another analysis type
        :param name: name of the analysis, str
        :return: results of the analysis
        """

        # This is specific to the Shelve backend, need to see if it can be removed.
        self.db.save(closedb=False)
        return self.db.get(name, self.mouse, self.date, self.run)
