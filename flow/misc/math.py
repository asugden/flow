from __future__ import division, print_function
from builtins import range

import numpy as np
import pandas as pd
import patsy
import re
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from scipy.stats import pearsonr
import seaborn as sns
import statsmodels.api as sm
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean


def subformula(formula, data):
    """
    Subset a dataframe to only include the essential elements for a formula.

    :param formula: formula pattern
    :return: subset of data frame
    """

    formula = re.sub('[\~|\+|\*|\(|\)|\|]', ' ', formula)
    keys = formula.split(' ')
    keys = list(set([k for k in keys if len(k) > 0 and not re.match('(\d|factor)', k)]))
    return data.loc[:, keys].copy()


def _mixed_keys(y, x, random_effects, categorical, continuous_random_effects):
    """
    Return the key names for x, y, and random_effects
    Parameters
    ----------
    y
    x
    random_effects

    Returns
    -------
    list of strs
        Keys to be extracted from dataframe

    """

    out = [y] + [v for v in x] + [v for v in categorical]
    res = [v for v in random_effects] + [v for v in continuous_random_effects]
    out += [vv for v in res for vv in v.split(':') if len(vv) > 0]
    return out


def mixed_effects_model(df, y, x=(), random_effects=(), categorical=(),
                        continuous_random_effects=(), family='gaussian',
                        dropzeros=False, nonlinear=False, R=False):
    """
    Create and run a mixed-effects general(ized) linear model.

    Parameters
    ----------
    df : Pandas DataFrame
    y : str
        The value in a dataframe to be fit
    x : tuple of str
        A tuple of fixed effects to fit
    random_effects : tuple of str
        A tuple of random effects to fit
    family : str {'gaussian', 'gamma'}
    link : str {'identity', 'log'}
    dropzeros : bool
        If true, remove all cases in which y is 0.
    nonlinear : bool
        If true, run a generalized linear model
        rather than a general linear model.
    R : bool
        Force the use of the R package

    Returns
    -------
    model

    """

    # Sanitize inputs
    if isinstance(x, basestring):
        x = (x, )
    if isinstance(categorical, basestring):
        categorical = (categorical, )
    if isinstance(random_effects, basestring):
        random_effects = (random_effects, )

    # Curate the data to a minimal size
    df_keys = _mixed_keys(y, x, random_effects, categorical, continuous_random_effects)
    for key in df_keys:
        if key not in df.keys():
            raise ValueError('Dataframe does not have column %s' % key)
    sub = df.loc[:, df_keys].copy()
    if dropzeros:
        sub.replace(0, np.nan, inplace=True)
    sub.dropna(inplace=True)

    # Convert columns of strings to integer factors
    for reff in random_effects:
        reff = reff.split(':')
        for v in reff:
            if len(v) > 0:
                sub[v], _ = pd.factorize(sub[v])

    # Convert categorical variables to strings
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    for c in categorical:
        vals = sub[c].unique()
        sub[c] = sub[c].replace({v:alpha[i] for i, v in enumerate(vals)})

    # Make formula
    form_pieces = ([v for v in x] + [v for v in categorical]
                   + ['(1|%s)'%s for s in random_effects if len(s) > 0]
                   + ['(%s)'%s for s in continuous_random_effects if len(s) > 0])
    formula = '%s ~ %s' % (y, ' + '.join(form_pieces))
    print(formula)

    if len(random_effects) == 0 and len(continuous_random_effects) == 0:
        y, X = patsy.dmatrices(formula, sub, return_type='dataframe')

        if family.lower() == 'gamma':
            linkfn = sm.families.links.log
            family = sm.families.Gamma(link=linkfn)
        elif family.lower() == 'gaussian' or family == 'normal':
            linkfn = sm.families.links.identity
            family = sm.families.Gaussian(link=linkfn)
        else:
            linkfn = sm.families.links.log
            family = sm.families.Poisson(link=linkfn)

        # import pdb;pdb.set_trace()

        model = sm.GLM(y, X, family=family)
        glm_results = model.fit()
        print(glm_results.summary2())
    else:
        rdf = pandas2ri.py2ri(sub)
        pandas2ri.activate()
        base = importr('base')
        # stats = importr('stats')
        afex = importr('afex')

        if family == 'gamma':
            family = 'Gamma'

        if nonlinear or family.lower() != 'gaussian':
            model = afex.mixed(formula=formula, data=rdf, method='PB', family=family)
        else:
            model = afex.mixed(formula=formula, data=rdf)

        print(base.summary(model))
        # print(model)

    return model

def glm(formula, df, family='gaussian', link='identity', dropzeros=True, r=False):
    """
    Apply a GLM using formula to the data in the dataframe data

    :param formula: text formulax
    :param df: pandas dataframe df
    :param family: string denoting what family of functions to use, gaussian, gamma, or poisson
    :param link: link function to use, identity or log
    :param dropzeros: replace zeros with nans if True
    :param r: use the R programming language's version if true
    :return: None
    """

    sub = subformula(formula, df)
    if dropzeros:
        sub.replace(0, np.nan, inplace=True)
    sub.dropna(inplace=True)

    if r:
        pattern = re.compile(r'(\(1|.+\))')
        for an in re.findall(pattern, formula):
            sub[an], _ = pd.factorize(sub[an])

        rdf = pandas2ri.py2ri(sub)
        pandas2ri.activate()
        base = importr('base')
        stats = importr('stats')

        model = stats.glm(formula, data=rdf)
        print(base.summary(model))

    else:
        y, X = patsy.dmatrices(formula, sub, return_type='dataframe')

        if link.lower() == 'log':
            linkfn = sm.families.links.log
        else:
            linkfn = sm.families.links.identity

        if family.lower() == 'gamma':
            family = sm.families.Gamma(link=linkfn)
        elif family.lower() == 'gaussian' or family == 'normal':
            family = sm.families.Gaussian(link=linkfn)
        else:
            family = sm.families.Poisson(link=linkfn)

        model = sm.GLM(y, X, family=family)
        glm_results = model.fit()
        print(glm_results.summary2())
        return model

def permutation_test(x, y, df, n=10000):
    """
    Run a permutation test on a value in a dataframe.

    Parameters
    ----------
    x : str
        name of x parameter
    y : str
        name of y parameter
    df : Pandas DataFrame
        Must contain columns x and y
    n : int
        Number of repetitions

    Returns
    -------
    int
        p-value
    """

    x = df[x].astype(np.float64).as_matrix()
    y = df[y].astype(np.float64).as_matrix()
    val = nancorr(x, y)[0]
    count = 0.0

    for i in range(n):
        rand = nancorr(x, np.random.permutation(y))[0]
        if rand > val:
            count += 1

    return count/n


def nancorr(v1, v2):
    """
    Return the correlation r and p value, ignoring positions with nan.

    :param v1: vector 1
    :param v2: vector 2, of length v1
    :return: (pearson r, p value)
    """

    # p value is defined as
    # t = (r*sqrt(n - 2))/sqrt(1-r*r)
    # where r is correlation coeffcient, n is number of observations, and the T is n - 2 degrees of freedom

    if len(v1) < 2 or len(v2) < 2:
        return (np.nan, np.nan)
    v1, v2 = np.array(v1), np.array(v2)
    nnans = np.bitwise_and(np.isfinite(v1), np.isfinite(v2))
    if np.sum(nnans) == 0:
        return (np.nan, np.nan)
    return pearsonr(v1[nnans], v2[nnans])


def smooth(x, window_len=5, window='flat'):
    """Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x
        the input signal
    window_len
        the dimension of the smoothing window; should be an odd integer
    window : {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}
        The type of window. Flat window will produce a moving average smoothing.

    Returns
    -------
        the smoothed signal

    Example
    -------

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError('Smooth only accepts 1 dimension arrays.')

    if x.size < window_len:
        raise ValueError('Input vector needs to be bigger than window size.')

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len/2-1):-(window_len/2)]


# ----------------------------------------------------------
# NaN operations

def nandivide(a, b):
    """
    Return nan if divide by 0 without error
    """

    if b == 0:
        return np.nan
    else:
        return float(a)/b

def weightcorr(x, y, w):
    """
    Return the weighted correlation of two vectors, x and y, and the weights w
    :param x:
    :param y:
    :return:
    """

    if len(x) != len(y) or len(x) != len(w):
        print('ERROR: Wrong lengths')

    # Weighted mean
    def m(x, w):
        return np.sum(x*w)/np.sum(w)

    # Weighted covariance
    def cov(x, y, w):
        return np.sum(w*(x - m(x, w))*(y - m(y, w)))/np.sum(w)

    # Weighted correlation
    def corr(x, y, w):
        return cov(x, y, w)/np.sqrt(cov(x, x, w)*cov(y, y, w))

    return corr(x, y, w)

def nancorr(v1, v2):
    """
    Return the correlation r and p value, ignoring positions with nan.
    :param v1: vector 1
    :param v2: vector 2, of length v1
    :return: (pearson r, p value)
    """

    # p value is defined as
    # t = (r*sqrt(n - 2))/sqrt(1-r*r)
    # where r is correlation coeffcient, n is number of observations, and the T is n - 2 degrees of freedom

    if len(v1) < 2 or len(v2) < 2: return (np.nan, np.nan)
    v1, v2 = np.array(v1), np.array(v2)
    nnans = np.bitwise_and(np.isfinite(v1), np.isfinite(v2))
    if np.sum(nnans) == 0: return (np.nan, np.nan)
    return pearsonr(v1[nnans], v2[nnans])

def nanpearson(v1, v2):
    """Same as nancorr."""
    return nancorr(v1, v2)

def nanspearman(v1, v2):
    """
    Return the correlation rho and p value, ignoring positions with nan.
    :param v1: vector 1
    :param v2: vector 2, of length v1
    :return: (spearman rho, p value)
    """

    v1, v2 = np.array(v1), np.array(v2)
    nnans = np.bitwise_and(np.isfinite(v1), np.isfinite(v2))
    return spearmanr(v1[nnans], v2[nnans])

def nanlinregress(v1, v2):
    """
    Return the linear regression accounting for nans
    :param v1:
    :param v2:
    :return:
    """

    v1, v2 = np.array(v1), np.array(v2)
    nnans = np.bitwise_and(np.isfinite(v1), np.isfinite(v2))
    return linregress(v1[nnans], v2[nnans])

def nannone(val):
    """
    Return value, converting None to np.nan for printing as floats
    :param val: value
    :return: float value
    """

    if val is None: return np.nan
    else: return val

def zeronone(val):
    """
    Return value, converting None to zero for printing as floats
    :param val: value
    :return: float value
    """

    if val is None: return 0.0
    else: return val

def emptynone(val):
    """
    Return value, converting None to empty lists
    :param val: value
    :return: listed value
    """

    if val is None:
        return []
    else:
        return val


# ----------------------------------------------------------
# Running statistics
# Based on http://www.johndcook.com/standard_deviation.html
def runstats(keep=0):
    """
    A function to return a new instance of RunningStats
    :return: instance of RunningStats
    """

    out = RunningStats(False, keep)
    return out


def runvecstats():
    """
    A function to return a new instance of RunningStats
    :return: instance of RunningStats
    """

    out = RunningStats(True)
    return out


class RunningStats(object):
    """
    Get a running mean and standard deviation from a huge dataset, so as not to save every possible point
    """
    def __init__(self, vec=False, keep=0):
        """
        Initialize with parameter keep. If keep > 0, a random
        sample of fraction keep will be kept for future statistics.
        :param keep: fraction (0-1) of examples to be kept
        """
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.vec = vec
        self.keep = keep
        self.subset = []

    def clear(self):
        """
        Start from zero again
        :return: None
        """
        self.n = 0
        self.subset = []

    def push(self, x):
        """
        Add a new point to the vector
        :param x: point to add
        :return: None
        """

        if not self.vec and isinstance(x, (list, tuple, np.ndarray)):
            for v in x:
                self.push(v)

        self.n += 1
        if (0 < self.keep < 1 and np.random.random() < self.keep) or self.keep == 1:
            self.subset.append(x)

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0 if not self.vec else np.zeros(len(x))

            if self.vec:
                self.old_m[np.invert(np.isfinite(x))] = 0
                self.new_s = np.copy(self.old_s)

        else:
            if self.vec:
                gx = np.isfinite(x)  # good x values
                self.new_m[gx] = self.old_m[gx] + (x[gx] - self.old_m[gx])/self.n
                self.new_s[gx] = self.old_s[gx] + (x[gx] - self.old_m[gx])*(x[gx] - self.new_m[gx])

            else:
                self.new_m = self.old_m + (x - self.old_m)/self.n
                self.new_s = self.old_s + (x - self.old_m)*(x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        """
        :return: The running mean from the data
        """
        return self.new_m if self.n else 0.0

    def variance(self):
        """
        :return: The running variance from the data
        """
        return self.new_s/(self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        """
        :return: The running standard deviation
        """
        return np.sqrt(self.variance())

    def standard_error(self):
        """
        Added by Arthur-- compute the running standard error
        :return: std err
        """
        return np.sqrt(self.variance())/np.sqrt(self.n)

    def samples(self):
        """
        Return reserved samples for use in statistics
        :return: vector, self.subset
        """
        return self.subset


def bootstrap_ci(
        data, ci=68, func=nanmean, axis=0, n_boot=10000, units=None):
    """Calculate a bootstrap confidence interval.

    Calculate the standard error of the mean (SEM) for a sample with ci=68 and
    func=mean/nanmean.

    Parameters
    ----------
    data : 1-d array
    ci : float on [0, 100]
        The desired confidence interval. The returned lower and upper bounds
        will contain ci% of the bootstrap statistics values.
    func : fn
        Desired statistic function.
    axis : int, optional
        Axis along which to resample data. If 'None', flatten array.
    n_bootstraps : int
    units : nd.array of same shape as data, optional
        Groupings of data. See seaborn.algorithms.bootstrap for details.

    Returns
    -------
    np.ndarray
        Lower and upper bounds of desired statistic.

    """

    boots = sns.algorithms.bootstrap(
        data, axis=axis, func=func, n_boot=n_boot, unit=units)

    confidence_intervals = sns.utils.ci(boots, which=ci, axis=axis)

    return confidence_intervals
