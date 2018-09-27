import numpy as np
import pandas as pd
import patsy
import re
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from scipy.stats import pearsonr
import statsmodels.api as sm


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