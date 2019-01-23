from __future__ import print_function
import community
from datetime import datetime
import networkx as nx
import numpy as np

from . import metadata as metadata
from . import paths, xday


def labels(mouse, day, minpred=0.01, minfrac=0.05):
    """
    Label cells by their responses to a GLM
    :param mouse: required for glm labels
    :param day: day, str
    :param minpred: the minimum variance predicted by all glm filters
    :param minfrac: the minimum fraction of glm variance explained by filters for each cell type
    :return: dict of labels
    """

    day = str(day)

    gdata = paths.getglm(mouse, day)

    odict, multiplexed = _glm(gdata, minpred, minfrac)
    odict['undefined'] = multiplexed == 0

    odict['multiplexed'] = multiplexed > 1
    categories = ['plus', 'neutral', 'minus', 'ensure', 'quinine', 'lick']
    for cat in categories:
        odict['%s-only' % cat] = np.bitwise_and(odict[cat], np.invert(odict['multiplexed']))
        odict['%s-multiplexed' % cat] = np.bitwise_and(odict[cat], odict['multiplexed'])

    odict['plus-ensure'] = np.bitwise_and(odict['plus'], np.bitwise_and(odict['ensure'], multiplexed == 2))
    odict['lick-ensure'] = np.bitwise_and(odict['lick'], odict['ensure'])
    odict['minus-quinine'] = np.bitwise_and(odict['minus'], np.bitwise_and(odict['quinine'], multiplexed == 2))

    return odict

def categorize(
        mouse,
        date,
        categories='reward quinine plus-only minus-only neutral-only lick'.split(' '),
        propagate=False,
        dayrange=(-3, 3),
        daydistance=(-999, 999),
        minpred=0.01,
        minfrac=0.05,
        andb=None):
    """
    Get the label for each pair of cells for a pair of days
    :param mouse: mouse name, str
    :param date: date, str
    :param categories: list of categories in order, such that if an individual cell has multiple categories,
    it is assigned to the first in this order
    :param propagate: If true, propagate labels in categories order from neighboring days
    :param dayrange: The number of recorded days to include (before, after)
    :param daydistance: The number of actual days to include (before, after)
    :param minpred: the minimum variance predicted by all glm filters
    :param minfrac: the minimum fraction of glm variance explained by filters for each cell type
    :return: a list of single labels per cell
    """

    if propagate:
        return _propagate_categories(mouse, date, categories, dayrange, daydistance, minpred, minfrac)

    else:
        lbls = labels(mouse, date, minpred, minfrac)
        # if andb is not None:
        #     lbls = addlabels(lbls, andb)

        ncells = len(lbls[lbls.keys()[0]])

        out = ['undefined' for i in range(ncells)]
        if 'cluster' in ''.join(categories):
            groupnames = cluster(andb, mouse, date)
            for key in groupnames:
                lbls[key] = groupnames[key]

        for c in range(len(out)):
            for cat1 in categories:
                if lbls[cat1][c]:
                    out[c] = cat1
                    break

        return out

def addlabels(out, andb):
    """
    Add labels in from analysis database
    :param out: dict of labels to add to
    :param andb: analysis database, indexed as dict
    :return: updated dict out
    """

    for cs in ['ensure', 'quinine']:
        out['%s-ttest'%cs] = andb['outcome-driven-%s'%cs]
        out['%s-anticipation-ttest'%cs] = andb['outcome-driven-anticipation-%s'%cs]
        out['%s-response-ttest'%cs] = andb['outcome-driven-response-%s'%cs]

    return out

def cluster(andb, mouse, date, cs='plus'):
    """
    Label by how many reward cells are within the cluster
    :param plusonly: cluster only within plus cells if true, otherwise all three stimuli
    :param max: maximum number of reward cells to use as a label
    :return: dict of boolean numpy arrays
    """

    out = {}
    for group in ['reward']:  # '['reward', 'ensure', 'quinine']:
        for nc in ['', '-0-1', '-nolick', '-0-1.5', '-0-1.5-decon']:
            nodes, edges = _nodeedges(andb, mouse, date, cs, nc)

            gx = nx.Graph()

            for c1 in nodes:
                gx.add_node(c1)

            for c1, c2, weight in edges:
                gx.add_edge(c1, c2, weight=weight)

            parts = community.best_partition(gx)

            catorder = ['ensure', 'quinine'] if group == 'ensure' else ['reward', 'quinine']
            lbls = categorize(mouse, date, catorder, propagate=False)
            groupcat = np.array([False if val != group else True for val in lbls], dtype=np.bool)
            groupcs = _count_communities(gx, nodes, parts, groupcat)

            for val in groupcs:
                out['%s%s-%s' % (group, nc, val)] = groupcs[val]

    return out

def category(lbls, categories, nonlabeled='undefined'):
    """
    Assign each cell to a category based on categories
    :param lbls: a dict of true/false labels
    :param categories: a list of categories
    :param nonlabeled: the category for those cells not labeled
    :return: a string assigned to each cell
    """

    out = [nonlabeled for val in range(len(lbls[categories[0]]))]

    for cat in categories:
        for c in range(len(lbls[cat])):
            if lbls[cat][c] and out[c] == nonlabeled:
                out[c] = cat

    return out

def _glm(simpglm, minpred=0.01, minfrac=0.05):
    """
    Return labels set by the glm
    :param simpglm: the cellgroups and variance explained straight from .simpglm file
    :param minpred: minimum deviation explained for the complete glm
    :param minfrac: minimum fraction of deviance explained by cell group
    :return: dict of labels
    """

    # Get cell groups and the sufficiency of each deviance explained
    groups = simpglm[0]
    groupdev = (simpglm[1][:, 1:].T*(simpglm[1][:, 0] >= minpred)).T
    groupdev = groupdev >= minfrac

    odict = {}
    for i, name in enumerate(groups):
        odict[name] = groupdev[:, i]

    # Multiplexed cells should respond not just to plus correct and plus error, but also to other types
    # Similarly, licking and ensure cannot be differentiated the way we're measuring them, so they can
    # also be combined.

    multiplexed = np.sum(groupdev, axis=1)

    if 'predict' in odict:
        odict['reward'] = np.bitwise_or(odict['ensure'], odict['predict'])
    else:
        odict['reward'] = odict['ensure']

    # Find out which are the true licking cells
    # behaviornames = [str(key[0]) for key in simpglm[2]['behaviornames'][0]]
    # for c in np.arange(len(odict['lick']))[odict['lick']]:
    #     if odict['ensure'][c]:
    #         bout = np.nansum(simpglm[2]['coefficients'][c,
    #                     np.array([i for i, x in enumerate(behaviornames) if x == 'lick_onsets'])])
    #         oth = np.nansum(simpglm[2]['coefficients'][c,
    #                                                     np.array([i for i, x in enumerate(behaviornames) if
    #                                                               x == 'lick_others'])])
    #         if oth == 0 or bout < 6:
    #             odict['lick'][c] = False

    return odict, multiplexed

def _propagate_categories(mouse, day1, categories, dayrange, daydistance, minpred, minfrac):
    """
    Propagate labels across time
    :param mouse:
    :param day1:
    :param categories:
    :param dayrange:
    :param daydistance:
    :param minpred:
    :param minfrac:
    :return:
    """

    dates = np.array(metadata.dates(mouse))
    if day1 not in dates:
        print('ERROR: Date not found')
        exit(0)

    d1 = np.argmax(dates == day1)
    dds = np.arange(len(dates)) - d1
    ddays = np.array([(datetime.strptime(str(day), '%y%m%d') - datetime.strptime(str(day), '%y%m%d')).days
                      for day in dates])

    backward = dates[(dds >= dayrange[0])
                     & (dds < 0)
                     & (ddays >= daydistance[0])][::-1]
    forward = dates[(dds <= dayrange[1])
                    & (dds > 0)
                    & (ddays <= daydistance[1])]

    lbl = categorize(mouse, day1, categories, minpred=minpred, minfrac=minfrac)
    lbl = _propagate_in_time(mouse, day1, backward, lbl, categories, minpred, minfrac)
    lbl = _propagate_in_time(mouse, day1, forward, lbl, categories, minpred, minfrac)

    return lbl


def _propagate_in_time(mouse, day1, dates, lbl, categories, minpred, minfrac):
    """
    Propagate labels forwards or backwards
    :param mouse:
    :param dates:
    :param lbl:
    :param categories:
    :param minpred:
    :param minfrac:
    :return:
    """

    xdaypre1, xdaypre2 = None, None
    for i, day2 in enumerate(dates):
        xdayp1, xdayp2 = xday.ids(mouse, day1, day2)

        if len(xdayp1) == 0 and len(xdayp2) == 0 and xdaypre1 is not None:
            t1, t2 = xday.ids(mouse, dates[i-1], day2)
            for p1, p2 in zip(xdaypre1, xdaypre2):
                if p2 in t1:
                    xdayp1.append(p1)
                    xdayp2.append(t2[np.argmax(t1 == p2)])
            xdayp1, xdayp2 = np.array(xdayp1), np.array(xdayp2)

        dlbl = categorize(mouse, day2, categories, propagate=False, minpred=minpred, minfrac=minfrac)
        for p1, p2 in zip(xdayp1, xdayp2):
            for cat in categories:
                if lbl[p1] == cat or dlbl[p2] == cat:
                    lbl[p1] = cat
                    break

        xdaypre1, xdaypre2 = xdayp1, xdayp2

    return lbl

def _nodeedges(andb, mouse, date, cs, nctext=''):
    """
    Get node labels and edge weights for days as noise correlations or spontanoues correlations (or combinations
    thereof)
    :param andb: analysis database instance
    :param mouse: mouse name, str
    :param date: date, str
    :param cs: stimulus name for noise correlations, blank for stimulus correlations
    :param combine: if True, combine stimulus and noise correlations
    :return:
    """

    if cs == 'spontaneous':
        corr = andb.get('spontaneous-correlation', mouse, date)
        nodes = np.arange(len(corr))
    else:
        corr = andb.get('noise-correlation%s-%s' % (nctext, cs), mouse, date)
        vdrive = andb.get('visually-driven-%s' % cs, mouse, date) > 50
        nodes = np.arange(len(vdrive))
        nodes = nodes[vdrive]

    corr[np.isnan(corr)] = -1
    corr[corr < 0] = 0

    edges = []

    for i, c1 in enumerate(nodes):
        for c2 in nodes[i+1:]:
            if corr[c1, c2] != np.nan and corr[c1, c2] > 0:
                edges.append((c1, c2, corr[c1, c2]))

    return nodes, edges

def _count_communities(gx, nodes, partitions, lbls):
    """
    Count those communities within each group
    :param gx: networkx graph
    :param nodes: list of nodes
    :param partitions: list of partitions by community analysis
    :param lbls: labels of a particular type, ensure or quinine
    :return:
    """

    ncommunities = len(np.unique([partitions.get(node) for node in gx.nodes()]))

    truenodes = np.zeros(len(lbls)) > 0
    rewards = {com: 0 for com in range(ncommunities)}
    for node in nodes:
        truenodes[node] = True
        if lbls[node]:
            rewards[partitions.get(node)] += 1

    out = {}
    for i in range(1, 2):
        out['cluster-%i' % i] = np.zeros(len(lbls)) > 0
        for node in nodes:
            if partitions.get(node) in rewards:
                if rewards[partitions.get(node)] >= i:
                    out['cluster-%i' % i][node] = True

        out['cluster-exclusive-%i' % i] = np.array([val for val in out['cluster-%i' % i]])
        out['cluster-exclusive-%i' % i][lbls] = False
        out['cluster-non-%i' % i] = np.bitwise_and(truenodes, np.invert(out['cluster-%i' % i]))
        out['cluster-exclusive-non-%i' % i] = np.bitwise_and(truenodes, np.invert(out['cluster-exclusive-%i' % i]))
        out['cluster-exclusive-non-%i' % i][lbls] = False

    return out
