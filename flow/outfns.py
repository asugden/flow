import numpy as np
from scipy import optimize
from scipy.spatial.distance import cosine as cosinedist
from scipy.stats import pearsonr, spearmanr, rankdata, linregress, t as tcdf

from . import glm, paths

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
    def m(x, w): return np.sum(x*w)/np.sum(w)

    # Weighted covariance
    def cov(x, y, w): return np.sum(w*(x - m(x, w))*(y - m(y, w)))/np.sum(w)

    # Weighted correlation
    def corr(x, y, w): return cov(x, y, w)/np.sqrt(cov(x, x, w)*cov(y, y, w))

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

def logzero(vec, replacement=0.0):
    """
    Return the log of a vector with all values that were <= 0 prior to log set to 0
    :param vec: vector
    :param replacement: replace zeros with this value
    :return: log of vector
    """

    vec = np.copy(vec)
    vec[vec < 0] = 0
    vec[vec > 0] = np.log(vec[vec > 0])
    vec[vec == 0] = replacement
    return vec

def partialcorr(x, y, zs, type='pearson', removenans=True):
    """
    Return the sample linear partial correlation coefficients between pairs of variables in X,
    controlling for remaining variables in X. Clone of Matlab's partialcorr
    :param mat: matrix of size N x M, N rows of M observations
    :return: rho (N x N matrix of corrs), p (N X N matrix of p values) P is two-tailed.
    """

    if removenans:
        nnans = np.bitwise_and(np.isfinite(x), np.bitwise_and(np.isfinite(y), np.isfinite(zs)))
        x = np.array(x)[nnans]
        y = np.array(y)[nnans]
        zs = np.array(zs)[nnans]

    x = np.array([x, y])
    zs = np.array(zs)
    n = np.shape(x)[1]

    # if np.sum(np.isnan(np.array([x, y]))):
    # 	print 'ERROR: nans in correlation'
    # 	exit(0)

    if type[0].lower() == 's':
        x[0, :] = rankdata(x[0, :])
        x[1, :] = rankdata(x[1, :])
        for i in range(np.shape(zs)[0]):
            zs[i, :] = rankdata(zs[i, :])

    zs1s = [np.ones(n)]
    if np.array(zs).ndim == 1: zs1s.append(zs)
    else:
        for col in zs:
            zs1s.append(col)
    zs1s = np.array(zs1s)

    dz = np.linalg.matrix_rank(zs)
    resid = x.transpose() - np.dot(zs1s.transpose(), np.linalg.lstsq(zs1s.transpose(), x.transpose())[0])

    tol = max(n, dz)*np.spacing(1)*np.sqrt(np.sum(np.square(np.abs(x)), axis=1))
    resid[:, np.sqrt(np.sum(np.square(np.abs(resid)), axis=0)) < tol] = 0

    coef = nancorr(resid[:, 0], resid[:, 1])[0]

    # Calculate the p values
    df = max(n - dz - 2, 0)

    t = coef/np.sqrt(1.0 - coef*coef)
    t *= np.sqrt(df)
    pval = 2*tcdf.cdf(-np.abs(t), df)

    return coef, pval

def residuals(x, y, zs):
    """
    Return the residuals of vectors x and y after controlling for correlations in zs. Clone of Matlab's partialcorr
    :param mat: matrix of size N x M, N rows of M observations
    :return: (residual x, residual y).
    """

    x = np.array([x, y])
    zs = np.array(zs)
    n = np.shape(x)[1]

    zs1s = [np.ones(n)]
    if np.array(zs).ndim == 1: zs1s.append(zs)
    else:
        for col in zs:
            zs1s.append(col)
    zs1s = np.array(zs1s)

    dz = np.linalg.matrix_rank(zs)
    resid = x.transpose() - np.dot(zs1s.transpose(), np.linalg.lstsq(zs1s.transpose(), x.transpose())[0])

    tol = max(n, dz)*np.spacing(1)*np.sqrt(np.sum(np.square(np.abs(x)), axis=1))
    resid[:, np.sqrt(np.sum(np.square(np.abs(resid)), axis=0)) < tol] = 0

    return resid[:, 0], resid[:, 1]

def rolling_window(a, window):
    """
    Rolling window makes a matrix out of a vector to be used by movingmax, movingmean, movingstdev
    :param a: vector
    :param window: window to use to expand
    :return: matrix of position-shifted vector copies
    """
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def movingmax(a, window):
    """
    Return the moving maximum/rolling maximum of a vector
    :param a: vector
    :param window: window size for moving maximum in frames
    :return: vector of size(a)
    """
    a = np.array(a)
    if a.ndim == 1: a = a.reshape((len(a), 1))

    beg = int(round((window - 0.5)/2.0))
    end = window - beg - 1
    beg = np.array([[np.nan]*np.shape(a)[0]]*beg).T
    end = np.array([[np.nan]*np.shape(a)[0]]*end).T

    a = np.concatenate([beg, a, end], axis=1)

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    strided = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    return np.nanmax(rolling_window(a, window), a.ndim)

def movingmean(a, window):
    """
    Return the moving mean/rolling mean of a vector
    :param a: vector
    :param window: window size for moving mean in frames
    :return: vector of size(a)
    """

    a = np.array(a)

    beg = int(round((window - 0.5)/2.0))
    end = window - beg - 1
    beg = np.array([[np.nan]*np.shape(a)[0]]*beg).T
    end = np.array([[np.nan]*np.shape(a)[0]]*end).T

    a = np.concatenate([beg, a, end], axis=1)

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    strided = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    return np.nanmean(rolling_window(a, window), a.ndim)

def movingmedian(a, window):
    """
    Return the moving mean/rolling mean of a vector
    :param a: vector
    :param window: window size for moving mean in frames
    :return: vector of size(a)
    """

    a = np.array(a)

    beg = int(round((window - 0.5)/2.0))
    end = window - beg - 1
    beg = np.array([[np.nan]*np.shape(a)[0]]*beg).T
    end = np.array([[np.nan]*np.shape(a)[0]]*end).T

    a = np.concatenate([beg, a, end], axis=1)

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    strided = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    return np.nanmedian(rolling_window(a, window), a.ndim)

def movingstdev(a, window):
    """
    Return the moving standard deviation/rolling standard deviation of a vector
    :param a: vector
    :param window: window size for moving standard deviation in frames
    :return: vector of size(a)
    """

    a = np.array(a)

    beg = int(round((window - 0.5)/2.0))
    end = window - beg - 1
    beg = np.array([[np.nan]*np.shape(a)[0]]*beg).T
    end = np.array([[np.nan]*np.shape(a)[0]]*end).T

    a = np.concatenate([beg, a, end], axis=1)

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    strided = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    return np.nanstd(rolling_window(a, window), a.ndim)

def printcorrelations(data, fields, stype='pearson', log=False, pcorr=''):
    """
    Print correlations and p-values of fields from data in a format that can be copied into a spreadsheet.
    :param data: a dict of lists/np arrays
    :param fields: a list of fields for which to print correlations
    :return:
    """

    title = 'Correlations'
    if log:
        title = 'Log Correlations'
    elif pcorr is not None and len(pcorr) > 0:
        title = 'Partial correlations to %s' % pcorr
    print('%s\t%s\tP Values (1 tail)\t%s' % (title, '\t'.join(fields[1:]), '\t'.join(fields[1:])))

    for i, key1 in enumerate(fields):
        if key1 != pcorr:
            line = [key1]
            for j, key2 in enumerate(fields[1:]):
                if key2 != pcorr:
                    if len(pcorr) > 0 and len(data[key2]) > 0 and len(data[key1]) > 0:
                        corr = partialcorr(data[key2], data[key1], data[pcorr], stype)[0]
                    elif stype == 'pearson':
                        corr = nanpearson(data[key2], data[key1])[0]
                    elif stype == 'spearman':
                        corr = nanspearman(data[key2], data[key1])[0]
                    else:
                        slope, intercept, r_value, p_value, std_err = nanlinregress(data[key1], data[key2])
                        corr = slope

                    if j < i: line.append(' ')
                    else: line.append('%.03f' % corr)
            line.append(key1)
            for j, key2 in enumerate(fields[1:]):
                if key2 != pcorr:
                    if len(pcorr) > 0 and len(data[key2]) > 0 and len(data[key1]) > 0:
                        corr = partialcorr(data[key2], data[key1], data[pcorr], stype)[1]/2.0
                    elif stype == 'pearson':
                        corr = nanpearson(data[key2], data[key1])[1]/2.0
                    elif stype == 'spearman':
                        corr = nanspearman(data[key2], data[key1])[1]/2.0
                    else:
                        slope, intercept, r_value, p_value, std_err = nanlinregress(data[key1], data[key2])
                        corr = p_value/2.0

                    if j < i: line.append(' ')
                    else: line.append('%.04f' % corr)
            print('\t'.join(line))

def xdaymatches(mouse, date1, date2):
    """
    Get the crossday id matches for date1 and date 2
    :param mouse: mouse, str
    :param date1: date1, str
    :param date2: date2, str
    :return: two vectors of rois that match between days
    """

    ids1 = metadata.readids(mouse, date1)
    ids2 = metadata.readids(mouse, date2)
    if len(ids1) == 0 or len(ids2) == 0: return [], []

    matchid1 = [i for i in range(len(ids1)) if ids1[i] in ids2]
    matchid2 = [ids2.index(ids1[mid]) for mid in matchid1]
    return np.array(matchid1), np.array(matchid2)

def combdicts(newd, origd, combby='append'):
    """
    Combine the entries of a new dictionary into an original dictionary using append
    :param newd: new dict
    :param origd: original dict to append to
    :param combby: combine by appending, adding
    :return: origd with newd info appended
    """

    for en in newd:
        if en not in origd:
            origd[en] = newd[en]
        else:
            for i in newd[en]:
                if combby[:3].lower() == 'app':
                    origd[en].append(i)
                elif combby[:3].lower() == 'add':
                    origd[en] += i
    return origd

def cellmask(andb, cs, mouse, day, cellthresh, maskfixednumber=False):
    """
    Return a numpy a mask for which cells to include
    :param andb: analysis database instance
    :param day: date, str
    :param cellthresh: cell threshold for cutoffs
    :param maskfixednumber: if true, threshold will mask a fixed number of cells. If false, will use drivenness p value
    :return: mask
    """

    if maskfixednumber:
        cmask = andb.get('stimulus-dff-0-2-%s'%cs, mouse, day)
        if cmask is None: return []
        cmask = np.argsort(cmask)[::-1]
        return cmask[:cellthresh]
    else:
        cmask = andb.get('visually-driven-%s'%cs, mouse, day)
        if cmask is None: return []
        else: return cmask > cellthresh

def concat(a, b, axis=0):
    """
    Use numpy concatenate iff a is already set. Otherwise set to b.
    :param a: vector or matrix
    :param b: vector or matrix
    :return: concatenated vector or matrix
    """

    if a == []:
        return b
    else:
        return np.concatenate([a, b], axis=axis)

def labels(andb, mouse, day, pval=0.05, useglm=True, minpred=0.01, minfrac=0.05, combine=False):
    """
    Label cells by their responses given a bonferroni corrected p value
    :param andb: analysis database instance
    :param mouse: required for glm labels
    :param day: day, str
    :param pval: p value to be bonferroni corrected
    :param glm: use output of glm if possible if true
    :param minpred: the minimum variance predicted by all glm filters
    :param minfrac: the minimum fraction of glm variance explained by filters for each cell type
    :param combine: if True, combine the outputs of both types of models
    :return: dict of labels
    """

    # if combine:
    #     gdata = paths.getglm(mouse, day)
    #     odict, multiplexed = labelglm(gdata, minpred, minfrac)
    #     odictt, multit = labelttest(andb, day, pval)
    #
    #     for group in odict:
    #         if group in odictt:
    #             if group == 'quinine':
    #                 # Combine two multiplexed with and for those cells that are uniplexed in both cases
    #                 # Then combine guaranteed uniplexed with and with the group
    #                 nonmultiplexedgroup = np.bitwise_and(odictt[group], np.bitwise_and(multiplexed == 0, multit == 1))
    #                 print(group, np.sum(np.bitwise_or(odict[group], nonmultiplexedgroup)) - np.sum(odict[group]))
    #                 odict[group] = np.bitwise_or(odict[group], nonmultiplexedgroup)
    #
    #     odict['undefined'] = np.bitwise_and(multiplexed == 0, multit == 0)
    # elif useglm:
    gdata = paths.getglm(mouse, day)
    odict, multiplexed = labelglm(gdata, minpred, minfrac)
    odict['undefined'] = multiplexed == 0
    # else:
    #     odict, multiplexed = labelttest(andb, day, pval)
    #     odict['undefined'] = multiplexed == 0

    odict['multiplexed'] = multiplexed > 1
    categories = ['plus', 'neutral', 'minus', 'ensure', 'quinine', 'lick']
    for cat in categories:
        odict['%s-only' % cat] = np.bitwise_and(odict[cat], np.invert(odict['multiplexed']))
        odict['%s-multiplexed' % cat] = np.bitwise_and(odict[cat], odict['multiplexed'])

    odict['plus-ensure'] = np.bitwise_and(odict['plus'], np.bitwise_and(odict['ensure'], multiplexed == 2))
    odict['minus-quinine'] = np.bitwise_and(odict['minus'], np.bitwise_and(odict['quinine'], multiplexed == 2))

    odict['ensure-vdrive'] = np.bitwise_and(odict['ensure'], andb.get('visually-driven-plus', mouse, day) > 50)
    odict['quinine-vdrive'] = np.bitwise_and(odict['quinine'], andb.get('visually-driven-minus', mouse, day) > 50)
    odict['ensure-vdrive'][odict['lick']] = False
    odict['quinine-vdrive'][odict['lick']] = False

    return odict

def labelglm(simpglm, minpred=0.01, minfrac=0.05):
    """
    Return labels set by the glm
    :param gdata: the cellgroups and variance explained straight from .simpglm file
    :param minpred: minimum deviation explained for the complete glm
    :param minfrac: minimum fraction of deviance explained by cell group
    :return: dict of labels
    """

    # Get cell groups and the sufficiency of each deviance explained
    groups = simpglm[0]
    groupdev = (simpglm[1][:, 1:].T*(simpglm[1][:, 0] >= minpred)).T
    groupdev = groupdev >= minfrac
    nrois = np.shape(simpglm[1])[0]

    out = np.zeros((len(groups), nrois), dtype=bool)
    odict = {}
    for i, name in enumerate(groups):
        odict[name] = groupdev[:, i]

    # Multiplexed cells should respond not just to plus correct and plus error, but also to other types
    # Similarly, licking and ensure cannot be differentiated the way we're measuring them, so they can
    # also be combined.

    multiplexed = np.sum(groupdev, axis=1)

    return odict, multiplexed

def labelttest(andb, mouse, day, pval):
    """
    Label cell groups by t-tests rather than by GLM
    :param andb: analysis database instance with mouse set
    :param day: day of analysis to retrieve
    :param pval: maximum p-value to accept
    :return:
    """
    names = ['plus-correct', 'plus-error', 'neutral-correct', 'neutral-error', 'minus-correct',
             'minus-error', 'ensure', 'licking', 'quinine']

    an = andb.get('modulated-by-%s'%names[0], mouse, day)
    i = 0
    while an is None and i < min(3, len(names)):
        i += 1
        an = andb.get('modulated-by-%s'%names[i], mouse, day)
    if an is None: return {}

    out = np.zeros((len(names), len(an)), dtype=bool)
    odict = {}
    np.seterr(under='ignore')
    for i, name in enumerate(names):
        an = andb.get('modulated-by-%s'%name, mouse, day)
        if an is None: an = np.ones(np.shape(out)[1])
        out[i, :] = an <= pval/len(names)
        odict[name] = out[i, :]

    # Multiplexed cells should respond not just to plus correct and plus error, but also to other types
    # Similarly, licking and ensure cannot be differentiated the way we're measuring them, so they can
    # also be combined.
    comb = 3
    outc = np.zeros((np.shape(out)[0] - comb, np.shape(out)[1]), dtype=bool)
    for c in range(comb):
        outc[c, :] = out[2*c, :] + out[2*c + 1, :]
    outc[comb:, :] = out[2*comb:, :]

    for i, name in enumerate(['plus', 'neutral', 'minus']):
        odict[name] = outc[i, :]

    multiplexed = np.sum(outc, axis=0)
    return odict, multiplexed

def eqdist(proto, event, cs='', usermvis=True):
    """
    Return the relative cosine distance to ensure and quinine given a vector of activity for a particular event
    :param proto: dict of prototypical vectors from protovectors (based on GLM)
    :param event: max or mean around replay or stimulus, to be compared to protovector
    :param cs: required if usermvis is set to true
    :param usermvis: use the visual-subtracted glm vector if available if true
    :return: ensure-quinine distance where values > 0 are closer to ensure, values < 0 closer to quinine
    """

    unitvec = event/np.nansum(np.abs(event))

    if usermvis and '%s-ensure' % cs in proto and '%s-quinine' % cs in proto:
        ensdist = cosinedist(proto['%s-ensure' % cs], unitvec)
        quidist = cosinedist(proto['%s-quinine' % cs], unitvec)
        return quidist - ensdist
    elif not usermvis and 'ensure' in proto and 'quinine' in proto:
        ensdist = cosinedist(proto['ensure'], unitvec)
        quidist = cosinedist(proto['quinine'], unitvec)
        return quidist - ensdist
    else:
        return np.nan

def uscsdist(proto, cs='', usermvis=False):
    """
    Return the relative cosine distance to ensure and quinine given a vector of activity for a particular event
    :param proto: dict of prototypical vectors from protovectors (based on GLM)
    :param event: max or mean around replay or stimulus, to be compared to protovector
    :param cs: required if usermvis is set to true
    :param usermvis: use the visual-subtracted glm vector if available if true
    :return: ensure-quinine distance where values > 0 are closer to ensure, values < 0 closer to quinine
    """

    if not usermvis:
        if cs == 'plus':
            return cosinedist(proto['ensure'], proto['plus'])
        elif cs == 'minus':
            return cosinedist(proto['quinine'], proto['minus'])
    else:
        fitfun = lambda vs, x: vs[0]*x
        errfun = lambda vs, x, y: fitfun(vs, x) - y

        if cs == 'plus':
            [vscalc, success] = optimize.leastsq(errfun, [0.5], args=(proto['ensure'], proto['plus']))
            return abs(vscalc[0])
        elif cs == 'minus':
            [vscalc, success] = optimize.leastsq(errfun, [0.5], args=(proto['quinine'], proto['minus']))
            return abs(vscalc[0])

    return 1

def edist(proto, event, cs='', usermvis=True):
    """
    Return the cosine distance to ensure given a vector of activity for a particular event
    :param proto: dict of prototypical vectors from protovectors (based on GLM)
    :param event: max or mean around replay or stimulus, to be compared to protovector
    :param cs: required if usermvis is set to true
    :param usermvis: use the visual-subtracted glm vector if available if true
    :return: ensure-quinine distance where values > 0 are closer to ensure, values < 0 closer to quinine
    """

    unitvec = event/np.nansum(np.abs(event))

    ensdist = np.nan
    if usermvis and '%s-ensure' % cs in proto:
        ensdist = cosinedist(proto['%s-ensure' % cs], unitvec)
    elif not usermvis and 'ensure' in proto:
        ensdist = cosinedist(proto['ensure'], unitvec)

    return ensdist

def qdist(proto, event, cs='', usermvis=True):
    """
    Return the cosine distance to quinine given a vector of activity for a particular event
    :param proto: dict of prototypical vectors from protovectors (based on GLM)
    :param event: max or mean around replay or stimulus, to be compared to protovector
    :param cs: required if usermvis is set to true
    :param usermvis: use the visual-subtracted glm vector if available if true
    :return: ensure-quinine distance where values > 0 are closer to ensure, values < 0 closer to quinine
    """

    unitvec = event/np.nansum(np.abs(event))

    ensdist = np.nan
    if usermvis and '%s-quinine' % cs in proto:
        ensdist = cosinedist(proto['%s-quinine' % cs], unitvec)
    elif not usermvis and 'quinine' in proto:
        ensdist = cosinedist(proto['quinine'], unitvec)

    return ensdist

def protovectors(mouse, date, trange=(0, 1), rectify=False, keep=[], err=-1, hz=None, rmvis=True):
    """
    Get the prototypical vectors, either from GLM basis sets or from means of stimuli
    :param mouse: mouse name, str
    :param date: date, str
    :param trange: time range in seconds, tuple
    :param rectify: whether to set values < 0 equal to 0 for glm basis set, boolean
    :param stims: output of stim events, dict of matrices ncells, nstimuli/onsets for each stimulus type
    :param err: trial errors, set to -1 for all trials, 0 for correct trials, 1 for error trials
    :param keep: a boolean vector of cells to keep
    :param hz: framerate of recordings, makes glm calculations faster
    :param rmvis: remove the visual component from ensure and quinine
    :return: unit vectors of each type
    """

    units = {}

    allunits = glm.unitvectors(mouse, date, trange, rectify, hz)
    if allunits is None or not allunits:
        return units

    for group in ['plus', 'neutral', 'minus']:
        if err < 0:
            if '%s_miss' % group in allunits:
                units[group] = (allunits['%s_correct' % group][keep] + allunits['%s_miss' % group][keep])/2.0
            else:
                units[group] = allunits['%s_correct' % group][keep]
        elif err == 0:
            units[group] = allunits['%s_correct' % group][keep]
        else:
            if '%s_miss' % group in allunits:
                units[group] = allunits['%s_miss' % group][keep]
            else:
                units[group] = allunits['%s_correct' % group][keep]

    for group in ['ensure', 'quinine', 'lick_onsets']:
        grname = group.split('_')[0]
        units[grname] = allunits[group][keep]

    for group in units:
        units[group] /= np.nansum(np.abs(units[group]))

    if rmvis:
        units = _remove_visual_components(units, ['plus', 'neutral', 'minus'])

    return units

def _remove_visual_components(proto, subtract):
    """
    Remove the visual component from ensure and quinine
    :param proto: dict of prototypical unit vectors
    :param subtract: stimulus to subtract
    :return: version of proto with stimulus subtracted
    """

    out = {}
    for key in proto: out[key] = np.copy(proto[key])

    for s in subtract:
        for egroup in proto:
            if egroup != s:
                # usvec = a*csvec + usvec_no_cs
                fitfun = lambda vs, x: vs[0]*x
                errfun = lambda vs, x, y: fitfun(vs, x) - y

                # with suppress_stdout_stderr:
                # seterr(invalid='ignore')
                [vscalc, success] = optimize.leastsq(errfun, [0.5], args=(proto[s], proto[egroup]))
                out['%s-%s' % (s, egroup)] = proto[egroup] - vscalc[0]*proto[s]
                out['%s-%s'%(s, egroup)] /= np.nansum(np.abs(out['%s-%s' % (s, egroup)]))

    return out

def checkday(andb, lpars, mouse, day):
    """
    Check whether a day fits the parameters
    :param andb: analysis database instance
    :param lpars: local parameters, including day-threshold, day-threshcomp, day-threshfn
    :param day: date, str, from metadata
    :return: True if day matches parameters or False
    """

    # Fix parameters
    if isinstance(lpars['day-threshold'], list):
        if len(lpars['day-threshold']) > 2:
            lpars['day-threshfn'], lpars['day-threshcomp'], lpars['day-threshold'] = lpars['day-threshold'][-3:]
        else:
            lpars['day-threshcomp'] = lpars['day-threshold'][-2]

    # Account for unset parameters
    if not lpars['day-threshold'] or lpars['day-threshfn'] == '': return True

    val = andb.get(lpars['day-threshfn'], mouse, day)
    if lpars['day-threshcomp'][0] == '>' or lpars['day-threshcomp'][0].lower() == 'g':
        return val >= lpars['day-threshold']
    else:
        return val < lpars['day-threshold']

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

class RunningStats:
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