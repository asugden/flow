from __future__ import division, print_function

import numpy as np
import pandas as pd
import patsy
import re
import statsmodels.api as sm
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


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


def _mixed_keys(y, x, random_effects, categorical, continuous_random_effects, offset):
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
    res = [v for v in random_effects] + [v for v in continuous_random_effects] + [v for v in offset]
    out += [vv for v in res for vv in v.split(':') if len(vv) > 0]
    return out


def mixed_effects_model(df, y, x=(), random_effects=(), categorical=(),
                        continuous_random_effects=(), family='gaussian',
                        dropzeros=False, nonlinear=False, offset=(), R=False, check_contrasts=False):
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
    if isinstance(x, str):
        x = (x, )
    if isinstance(categorical, str):
        categorical = (categorical, )
    if isinstance(random_effects, str):
        random_effects = (random_effects, )
    if isinstance(offset, basestring):
        offset = (offset, )

    use_R = True if (len(random_effects) > 0
                     or len(continuous_random_effects) > 0) else False

    # Curate the data to a minimal size
    df_keys = _mixed_keys(y, x, random_effects, categorical, continuous_random_effects, offset)
    for key in df_keys:
        if key not in df.keys():
            raise ValueError('Dataframe does not have column %s' % key)
    sub = df.loc[:, df_keys].copy()
    if dropzeros:
        sub.replace(0, np.nan, inplace=True)
    sub.dropna(inplace=True)

    for key in offset:
       sub[key] = np.log(sub[key])

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
                   #+ ['offset(%s)'%s for s in offset if len(offset) > 0 and use_R])
    formula = '%s ~ %s' % (y, ' + '.join(form_pieces))
    print(formula)

    if not use_R:
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

        model = sm.GLM(y, X, family=family, offset=None if len(offset) == 0 else sub[offset[0]])
        glm_results = model.fit()
        print(glm_results.summary2())
    else:
        rdf = pandas2ri.py2ri(sub)
        pandas2ri.activate()
        base = importr('base')
        # stats = importr('stats')
        afex = importr('afex')
        # multcomp = importr('multcomp')

        if family == 'gamma':
            family = 'Gamma'

        #print(sub)

        if nonlinear or family.lower() != 'gaussian':
            if offset != ():
                rdfoff = pandas2ri.py2ri(sub[offset[0]])

                model = afex.mixed(formula=formula, data=rdf, method='PB', family=family, offset=rdfoff, check_contrasts=False)
                # print(base.summary(multcomp.glht(model)))

            else:
                model = afex.mixed(formula=formula, data=rdf, method='PB', family=family)
        else:
            model = afex.mixed(formula=formula, data=rdf)

        print(base.summary(model))
        print(model)

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
        elif link.lower() == 'inverse':
            linkfn = sm.families.links.inverse_power
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
