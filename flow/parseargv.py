from copy import copy, deepcopy
import numpy as np
import os
import os.path as opath

# This dependency should probably be removed if possible.
# For now moved inside functions that need it
# from pool import database

from . import config
from . import metadata
from . import paths
from .classifier import classify
from .misc import loadmat
from .misc import Parser

"""
Parse command-line inputs into easily readable versions to be used to 
select a classifier.
"""


class CommandInputs:
    def __init__(self, args):
        self.t = args
        if len(args) < 4:
            print('Program should be run with [mouse] [date] [training,days] <comparison-run> {comparison rather than training = true} {randomize}')
            exit(0)

    def parse(self):
        """
        Parse the command-line inputs saved in self.t
        """

        randomize = False
        randomizationType = 'first-second-order'
        args = []
        for a in self.t:
            if a[:6].lower() == 'random':
                randomize = True
            elif a.lower() == 'circshift':
                randomizationType = 'circshift'
            elif a[:6].lower() == 'median':
                randomizationType = 'median'
            elif a.lower() == 'cells':
                randomizationType = 'cells'
            elif a.lower() == 'circshift-cells':
                randomizationType = 'median'
            else:
                args.append(a)

        out = config.default()
        out['mouse'] = args[1]
        out['training-date'] = args[2]
        out['comparison-date'] = args[2]
        out['training-runs'] = [int(i) for i in args[3].split(',')]
        out['comparison-run'] = int(args[4]) if len(args) > 4 else 0
        out['randomize'] = randomize
        out['randomization-type'] = randomizationType

        return out

def parse(args):
    """
    Parse command-line arguments and return.
    """

    ci = CommandInputs(args)
    return ci.parse()


def getmetadata(args):
    """
    Get the metadata specific to a particular mouse, date, and run.
    """

    date = args['date'] if 'date' in args else args['training-date']
    mouse = args['mouse']

    if 'noauto' in args:
        return args
    if mouse not in metadata.spontaneous:
        return args

    md = metadata.spontaneous[mouse]

    out = {p: args[p] for p in args}

    for group in md:
        if str(date) in md[group]:
            if 'training-runs' not in args:
                out['training-runs'] = md[group]['train']
            if 'training-other-running-runs' not in args:
                out['training-other-running-runs'] = md[group]['running']

    return out

def parsekv(args, defaults=True):
    """
    Parse command-line arguments and return.
    """

    argstr = ' '.join(args)
    p = Parser.Parser()
    pars = p.keyvals(argstr)

    # Don't add metadata or default parameters
    if not defaults:
        return pars

    pars = getmetadata(pars)

    # Format correctly
    fpars = config.default()
    for p in pars:
        if p in fpars:
            fpars[p] = pars[p]

        if p == 'date':
            fpars['training-date'] = str(pars[p])
            fpars['comparison-date'] = str(pars[p])
        elif p == 'run':
            fpars['comparison-run'] = pars[p]
        elif p == 'training-date':
            fpars[p] = str(pars[p])
        elif p == 'comparison-date':
            fpars[p] = str(pars[p])
        elif p == 'training-other-running-runs':
            if isinstance(pars[p], int):
                fpars[p] = [pars[p]]

    return fpars

def go(args, defaults={}, classifier=False, trace=False, force=False):
    """
    Extract local parameters, classifier parameters, and account
    for a request for default parameters.
    :param args: arguments from the command line
    :param defaults: default arguments for local parameters
    :param classifier: set to true if a classifier should be returned
    :param trace: set to true if a trace should be returned
    :param force: set to true to force classification
    :return: classifier pars, local pars, optionally classifier and trace
    """

    # Print defaults if desired
    defaults['default'] = False
    defaults['defaults'] = False
    defaults['def'] = False
    lpars = extractkv(args, defaults)
    if lpars['default'] or lpars['defaults'] or lpars['def']:
        keys = sorted([key for key in defaults if key[:3] != 'def'])
        print('DEFAULT PARAMETERS:')
        for key in keys:
            print '\t%s:' % (key),
            print defaults[key]
    pars = parsekv(args)

    # Account for the various output possibilities
    if not classifier:
        if not trace:
            return (pars, lpars)
        else:
            return (pars, lpars, trace2p(pars))
    else:
        randomize = random(args)
        cf = classifier(pars, randomize, force)
        if cf == {} and not trace: return ({}, lpars, {})
        elif cf == {} and trace: return ({}, lpars, {}, trace2p(pars))
        elif not trace: return (pars, lpars, cf)
        else: return (pars, lpars, cf, trace2p(pars))

def extractkv(args, defaults=None):
    """
    Extract the parameters of defaults from the argstring args.
    """
    if defaults is None:
        defaults = {}
    # Copy over the default values
    out = {}
    for d in defaults:
        out[d] = defaults[d]

    # And update the defaults
    argstr = ' '.join(args)
    p = Parser.Parser()
    pars = p.keyvals(argstr)
    for p in pars:
        if p in out:
            out[p] = pars[p]

    return out


def parseargs(args, defaults=None, limit_to_defaults=False):
    """
    Combine arguments from command-line and defaults.

    Parameters
    ----------
    args : Namespace
        Parsed args from argparse parse.
    defaults : dict
        Dictionary of default values.

    Returns
    -------


    """
    if defaults is None:
        defaults = {}

    out = deepcopy(defaults)

    for key, val in vars(args).iteritems():
        if not limit_to_defaults or key in defaults:
            out[key] = val
    return out


def parseclass(args, trace=False, force=True):
    """
    Parse input arguments, get the resulting classifier and optionally
    the trace2p file. If force is true, it will classify as desired and
    return the output.
    """

    pars = parsekv(args)
    randomize = random(args)

    cf = classifier(pars, randomize, force)
    if cf == {} and not trace: return ({}, {})
    elif cf == {} and trace: return ({}, {}, {})
    elif not trace: return (pars, cf)
    else: return (pars, cf, trace2p(pars))

def classifiermdr(mouse, date, run, randomize='', force=True):
    """
    Return a classifier output given a mouse, date, run, and the current settings file.
    :param mouse: mouse name, str
    :param date: date, str (yymmdd)
    :param run: run, int
    :param randomize: randomization string
    :param force: create a new classifier if possible
    :return: classifier output or None
    """

    mousemd = metadata.mdr(mouse, date, run)
    args = parsekv([
        '-mouse', mouse,
        '-date', date,
        '-comparison-run', str(run),
        '-training-runs', ','.join([str(i) for i in mousemd['train']]),
        '-training-other-running-runs', ','.join([str(i) for i in mousemd['running']])
    ])

    return classifier(args, randomize, force)

def classifier(pars, randomize='', force=False):
    """
    Get the most recent classifier of the appropriate type.
    """

    path = paths.output(pars)
    fs = os.listdir(path)[::-1]
    out = ''

    # Change what you open whether real or random
    if len(randomize) == 0:
        for f in fs:
            if len(out) == 0:
                if f[:4] == 'real':
                    out = opath.join(path, f)
    else:
        for f in fs:
            if len(out) == 0:
                if f[:4] == 'rand' and f[5:5 + len(randomize)] == randomize:
                    out = opath.join(path, f)

    if len(out) == 0 and not force: return {}
    elif len(out) == 0 and force:
        classify.classify(pars, randomize)
        return classifier(pars, randomize, False)
    else:
        return loadmat(out)

def classifiers(pars, randomize='', minclassifiers=1, check=False):
    """
    Get multiple copies of classifiers. For example, with randomization,
    one would want multiple randomized copies to double-check one's work
    """

    path = paths.output(pars)
    fs = os.listdir(path)[::-1]
    out = []

    # Change what you open whether real or random
    if len(randomize) == 0:
        for f in fs:
            if f[:4] == 'real':
                out.append(opath.join(path, f))
    else:
        for f in fs:
            if f[:4] == 'rand' and f[5:5 + len(randomize)] == randomize:
                out.append(opath.join(path, f))

    # Now that we know how many randomized classifications have been
    # made, we can make as many more as necessary
    if len(out) < minclassifiers:
        if check and len(randomize) > 0:
            t2p = trace2p(pars)
            if np.sum(t2p.inactivity()) < 100: return []
        # for i in range(minclassifiers - len(out)):
        #     classify.classify(pars, randomize)
        if minclassifiers - len(out) > 1:
            print('multiclassify ', minclassifiers - len(out))
            classify.multiclassify(pars, randomize, minclassifiers - len(out))
        else:
            classify.classify(pars, randomize)

        return classifiers(pars, randomize, False)

    # Load and return all classifier results
    loaded = []
    for p in out:
        cls = loadmat(p)
        loaded.append(cls)
    return loaded

def trace2p(pars):
    """
    Get the trace2p file for the given input parameters.
    """

    t2p = paths.gett2p(pars['mouse'], pars['comparison-date'], pars['comparison-run'])
    return t2p

def random(args):
    """
    Should the output be randomized?
    """

    argstr = ' '.join(args)
    p = Parser.Parser()
    pars = p.keyvals(argstr)
    out = ''

    if 'randomize' in pars:
        options = ['cells', 'circshift', 'median', 'cells-circshift', 'first-second-order']
        if pars['randomize'].lower() in options:
            out = pars['randomize'].lower()

    return out

class RunSorter():
    def __init__(self, args, classifier=True, trace=False, force=False, multiclassifiers=False, minclassifiers=1, all=False):
        """
        RunSorter class sorts a series of days and has a "next" method
        to call for the next sorted run.
        :param args: arguments from the command line
        :param classifier: set to True if a classifier should be returned
        :param trace: set to True if a trace2p file should be returned
        :param force: set to True if classification should be forced
        :param multiclassifiers: return multiple classifers as output
        :param minclassifiers: minimum number of classifiers to return
        :param all: allow all kinds of days, not just spontaneous ones
        (similar to forc`e)
        """

        self.retclassifier = classifier
        self.rett2p = trace
        self.force = force
        self.multiclassifiers = multiclassifiers
        self.minclassifiers = minclassifiers

        kvargs = parsekv(args, False)
        self.cmdargs = deepcopy(kvargs)

        self._daylimit, self.andb = None, None
        self._day_threshold(kvargs)

        self.md = metadata.sortedall() if all else metadata.sortedspontaneous()

        self.randomize = '' if 'randomize' not in kvargs else kvargs['randomize']
        self._prune_metadata(kvargs)

        self.index = -1

    def _day_threshold(self, kvargs):
        """
        Check for a day-threshold limitation
        :param kvargs: key-value arguments from parsekv
        :return: None
        """
        from pool import database
        if 'day-analysis' in kvargs:
            self.andb = database.db()
            stringed = ''.join([str(v) for v in kvargs['day-analysis'][1:]])
            if '[' not in stringed or ']' not in stringed:
                print 'ERROR: Variable required. Use square brackets, [], to surround variables.'
                exit(0)

            self._daylimit = 'limit = %s' % \
                (stringed.replace('[', ' self.andb.get(\''))

    def _prune_metadata(self, kvargs):
        """
        Cut out all animals/days/days/groups that do not match
        """

        newmd = []

        for mouse, date, run, group in self.md:
            if (matchvalorlist(kvargs, 'mouse', mouse, True) and
                    matchvalorlist(kvargs, 'date', int(date)) and
                    matchvalorlist(kvargs, 'run', int(run)) and
                    matchvalorlist(kvargs, 'group', group) and
                    metadata.checkreversal(mouse, date, kvargs, 'reversal')):

                daypass = True
                if self._daylimit is not None:
                    limit = True
                    exec(self._daylimit.replace(']', '\', %i)' % int(date)))
                    if not limit:
                        daypass = False

                if daypass:
                    newmd.append((mouse, date, run, group))

        self.md = newmd

    def next(self):
        """
        Advance to the next index. Meant to be used in while loop
        :return: True if another index exists, else false
        """

        # Check that we haven't gone past the last index
        self.index += 1
        if self.index < len(self.md):
            # First get the arguments
            mouse, date, run, group = self.md[self.index]
            mousemd = metadata.mdr(mouse, date, run)
            self.nextargs = parsekv([
                '-mouse', mouse,
                '-date', date,
                '-comparison-run', str(run),
                '-training-runs', ','.join([str(i) for i in mousemd['train']]),
                '-training-other-running-runs', ','.join([str(i) for i in mousemd['running']])
            ])

            # Only pass if classifier exists, save classifier
            if self.retclassifier and not self.multiclassifiers:
                cf = classifier(self.nextargs, self.randomize, self.force)

                # Recursively move forwards if classifier does not exist
                if cf == {}: return self.next()

                # Save values for returning
                self.nextclassifier = cf

            if self.multiclassifiers:
                cfs = classifiers(self.nextargs, self.randomize, self.minclassifiers)

                # Recursively move forwards if classifier does not exist
                if len(cfs) == 0: return self.next()

                # Save values for returning
                self.nextclassifier = cfs

            # Save t2p file
            if self.rett2p:
                self.nextt2p = trace2p(self.nextargs)

            return True
        else: return False

    def back(self):
        """
        Return to the previous position
        """
        self.index -= 2
        self.next()

    def get(self):
        """
        Return all of the requested values
        :return: various, but all requested
        """

        # Check that the indexes are correct
        if self.index >= len(self.md) or self.index < 0: return ()

        out = [self.md[self.index], self.nextargs]
        if self.retclassifier or self.multiclassifiers: out.append(self.nextclassifier)
        if self.rett2p: out.append(self.nextt2p)

        return tuple(out)

    def metadata(self):
        """
        :return: the metadata of the specific index
        """
        return self.md[self.index]

    def args(self):
        """
        :return: arguments/settings for a specific classifier
        """
        return self.nextargs

    def name(self, cs=False):
        """
        Return a name based on the command line arguments passed.
        Specifically cares about mouse and date
        :param cs: if cs, include the cs values if passed in command line
        :return:
        """

        out = ''
        if 'mouse' in self.cmdargs:
            if isinstance(self.cmdargs['mouse'], str):
                out += self.cmdargs['mouse']
            else:
                out += ','.join(sorted(self.cmdargs['mouse']))
        else: out += 'allmice'

        if 'group' in self.cmdargs:
            out += '-%s'%str(self.cmdargs['group'])

        if 'training-date' in self.cmdargs: out += '-%s'%str(self.cmdargs['training-date'])
        elif 'date' in self.cmdargs: out += '-%s'%str(self.cmdargs['date'])

        if 'comparison-run' in self.cmdargs:
            if isinstance(self.cmdargs['comparison-run'], int): out += '-%i'%self.cmdargs['comparison-run']
            else: out += '-' + ','.join([str(i) for i in self.cmdargs['comparison-run']])

        if cs and 'cs' in self.cmdargs:
            out += '-'
            if 'plus' in self.cmdargs['cs']: out += 'p'
            if 'neutral' in self.cmdargs['cs']: out += 'n'
            if 'minus' in self.cmdargs['cs']: out += 'm'

        return out

    def classifier(self):
        """
        :return: the classifier for this index
        """
        return self.nextclassifier

    def trace(self):
        """
        :return: the trace2p file for this index
        """
        return self.nextt2p

    def reset(self):
        """
        Reset the index back to the first position
        """
        self.index = -1


def matchvalorlist(args, key, val, strict=False):
    """
    Match a parameter to a value or list from kvargs.

    :param arg: argument, can be value or list
    :param val: value to be matched
    :param strict: force a string match to be perfect
    :return: true or false, true if matched

    """
    # First, if the key is not present, everything matches
    if key not in args:
        return True

    # If the key is present, check the type
    if isinstance(args[key], list) or isinstance(args[key], tuple):
        return val in args[key]
    elif isinstance(args[key], str):
        if strict:
            return val.lower() == args[key].lower()
        else:
            # if val.lower() in args[key].lower(): return True
            # elif args[key].lower() in val.lower(): return True
            # else: return False
            return val.lower() in args[key].lower() or \
                args[key].lower() in val.lower()
    else:
        return val == args[key]


class DaySorter():
    def __init__(self, args, classifier=True, traces=False, force=False, multiclassifiers=False, minclassifiers=1):
        """
        RunSorter class sorts a series of days and has a "next" method
        to call for the next sorted run.
        :param args: arguments from the command line
        :param classifier: set to True if a classifier should be returned
        :param trace: set to True if a trace2p file should be returned
        :param force: set to True if classification should be forced
        :param multiclassifiers: return multiple classifers as output
        :param minclassifiers: minimum number of classifiers to return
        (similar to force)
        """

        self.retclassifier = classifier
        self.rett2p = traces
        self.force = force
        self.multiclassifiers = multiclassifiers
        self.minclassifiers = minclassifiers

        kvargs = parsekv(args, False)
        self.cmdargs = deepcopy(kvargs)

        self._daylimit, self.andb = None, None
        self._day_threshold(kvargs)

        self.md = metadata.sortedspontaneous()
        self.randomize = '' if 'randomize' not in kvargs else kvargs['randomize']
        self._prune_metadata(kvargs)

        self.index = -1

    def _day_threshold(self, kvargs):
        """
        Check for a day-threshold limitation
        :param kvargs: key-value arguments from parsekv
        :return: None
        """
        from pool import database
        if 'day-analysis' in kvargs:
            self.andb = database.db()
            stringed = ''.join([str(v) for v in kvargs['day-analysis'][1:]])
            if '[' not in stringed or ']' not in stringed:
                print 'ERROR: Variable required. Use square brackets, [], to surround variables.'
                exit(0)

            self._daylimit = 'limit = %s' % (stringed.replace('[', ' self.andb[\''))
            self._daylimit = self._daylimit.replace(']', '\']')

    def _prune_metadata(self, kvargs):
        """Cut out all animals/days/days/groups that do not match."""

        newmd = []
        for mouse, date, run, group in self.md:
            if (matchvalorlist(kvargs, 'mouse', mouse, True) and
                    matchvalorlist(kvargs, 'date', int(date)) and
                    matchvalorlist(kvargs, 'group', group) and
                    metadata.checkreversal(mouse, date, kvargs, 'reversal')):

                daypass = True
                if self._daylimit is not None:
                    self.andb.md(mouse, date)
                    limit = True
                    exec(self._daylimit)
                    if not limit:
                        daypass = False

                if daypass:
                    if len(newmd) > 0 and newmd[-1][0] == mouse and \
                            newmd[-1][1] == date:
                        newmd[-1][2].append(run)
                    else:
                        newmd.append([mouse, date, [run], group])

        self.md = newmd

    def next(self):
        """
        Advance to the next index. Meant to be used in while loop
        :return: True if another index exists, else false
        """

        # Check that we haven't gone past the last index
        self.index += 1
        if self.index < len(self.md):
            # First get the arguments
            mouse, date, runs, group = self.md[self.index]

            self.nextclassifier = []
            self.nextt2p = []
            self.nextargs = []

            for run in runs:
                mousemd = metadata.mdr(mouse, date, run)
                self.nextargs.append(deepcopy(parsekv([
                    '-mouse', mouse,
                    '-date', date,
                    '-comparison-run', str(run),
                    '-training-runs', ','.join([str(i) for i in mousemd['train']]),
                    '-training-other-running-runs', ','.join([str(i) for i in mousemd['running']])
                ])))

                # Only pass if classifier exists, save classifier
                if self.retclassifier and not self.multiclassifiers:
                    cf = classifier(self.nextargs[-1], self.randomize, self.force)

                    # Recursively move forwards if classifier does not exist
                    if cf == {}: return self.next()

                    # Save values for returning
                    self.nextclassifier.append(cf)

                if self.multiclassifiers:
                    cfs = classifiers(self.nextargs[-1], self.randomize, self.minclassifiers)

                    # Recursively move forwards if classifier does not exist
                    if len(cfs) == 0: return self.next()

                    # Save values for returning
                    self.nextclassifier.append(cfs)

                # Save t2p file
                if self.rett2p:
                    self.nextt2p.append(trace2p(self.nextargs[-1]))

            return True
        else: return False

    def get(self):
        """
        Return all of the requested values
        :return: various, but all requested
        """

        # Check that the indexes are correct
        if self.index >= len(self.md) or self.index < 0: return ()
        out = [self.md[self.index], self.nextargs]
        if self.retclassifier or self.multiclassifiers: out.append(self.nextclassifier)
        if self.rett2p: out.append(self.nextt2p)

        return tuple(out)

    def metadata(self):
        """
        :return: the metadata of the specific index
        """
        return self.md[self.index]

    def args(self):
        """
        :return: arguments/settings for a specific classifier
        """
        return self.nextargs

    def name(self, cs=False):
        """
        Return a name based on the command line arguments passed.
        Specifically cares about mouse and date
        :param cs: if cs, include the cs values if passed in command line
        :return:
        """

        out = ''
        if 'mouse' in self.cmdargs:
            if isinstance(self.cmdargs['mouse'], str):
                out += self.cmdargs['mouse']
            else:
                out += ','.join(sorted(self.cmdargs['mouse']))
        else:
            out += 'allmice'

        if 'group' in self.cmdargs:
            out += '-%s'%str(self.cmdargs['group'])

        if 'training-date' in self.cmdargs: out += '-%s'%str(self.cmdargs['training-date'])
        elif 'date' in self.cmdargs: out += '-%s'%str(self.cmdargs['date'])

        if 'reversal' in self.cmdargs:
            if self.cmdargs['reversal'] == 'pre' or self.cmdargs['reversal'] == 'post':
                out += '-rev%s' % self.cmdargs['reversal']

        if cs and 'cs' in self.cmdargs:
            out += '-'
            if 'plus' in self.cmdargs['cs']: out += 'p'
            if 'neutral' in self.cmdargs['cs']: out += 'n'
            if 'minus' in self.cmdargs['cs']: out += 'm'

        return out

    def classifier(self):
        """
        :return: the classifier for this index
        """
        return self.nextclassifier

    def trace(self):
        """
        :return: the trace2p file for this index
        """
        return self.nextt2p

    def reset(self):
        """
        Reset the index back to the first position
        """
        self.index = -1

def sortedruns(args, classifier=True, trace=False, force=False, multiclassifiers=False, minclassifiers=1, allruns=False):
    """
    Return a class that can call "next" of sorted spontaneous
    days that match args.
    :param args: arguments from the command line matching settings
    :param classifier: set to True if a classifier should be returned
    :param trace: set to True if a trace2p file should be returned
    :param force: set to True if classification should be forced
    :param multiclassifiers: return multiple classifers as output
    :param minclassifiers: minimum number of classifiers to return
    (similar to force)
    :return: a class ready to be iterated over with next
    """

    out = RunSorter(args, classifier, trace, force, multiclassifiers, minclassifiers, allruns)
    return out

def sorteddays(args, classifier=True, trace=False, force=False, multiclassifiers=False, minclassifiers=1):
    """
    Return a class that can call "next" of sorted spontaneous
    days that match args.
    :param args: arguments from the command line matching settings
    :param classifier: set to True if a classifier should be returned
    :param trace: set to True if a trace2p file should be returned
    :param force: set to True if classification should be forced
    :param multiclassifiers: return multiple classifers as output
    :param minclassifiers: minimum number of classifiers to return
    (similar to force)
    :return: a class ready to be iterated over with next
    """

    out = DaySorter(args, classifier, trace, force, multiclassifiers, minclassifiers)
    return out

def listify(val):
    """
    Make sure that val is a list to be iterated over
    :param val: list, tuple, numpy ndarray, int, float, or str
    :return: list
    """

    if isinstance(val, tuple):
        val = list(val)
    elif isinstance(val, np.ndarray):
        val = val.tolist()
    elif isinstance(val, str) or isinstance(val, int) or isinstance(val, float):
        val = [val]

    return val