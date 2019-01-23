from builtins import range
from builtins import object
import community
import networkx as nx
import numpy as np


class NCGraph(object):
    def __init__(self, nodes, corr, limits=None):
        """
        Generate a graph of noise correlations using networkx
        :param nodes: can be int or a list defining the range
        :param corr: correlations that define edges
        :param limits: limitations on which nodes to include
        """

        self.ncells = nodes if isinstance(nodes, int) else len(nodes)
        self.limits = limits
        self.labels = None

        self.nodes, self.edges = nodes_edges(nodes, corr, limits)

        # Add nodes and edges
        self.gx = nx.Graph()
        for c1 in self.nodes:
            self.gx.add_node(c1)
        for c1, c2, weight in self.edges:
            self.gx.add_edge(c1, c2, weight=weight)

    def connectivity(self):
        """
        Return the networkx clustering-- the geometric mean for each node of the associated edges.
        :return: numpy array of values
        """

        cluster = nx.clustering(self.gx, None, weight='weight')

        out = np.full(self.ncells, np.nan)
        for i, c1 in enumerate(self.nodes):
            if cluster[c1] > 0:
                out[c1] = cluster[c1]

        return out

    def communities(self):
        """
        Return the community ID of each node, using Louvain community analysis
        :return: numpy array of values
        """

        # Run community detection
        # Instructions from https://blog.dominodatalab.com/social-network-analysis-with-networkx/
        # Louvain community detection code from https://github.com/taynaud/python-louvain/
        parts = community.best_partition(self.gx)

        out = np.full(self.ncells, np.nan)
        for node in self.nodes:
            out[node] = parts.get(node)

        return out

    def ncommunities(self):
        """
        Return the number of communities of graph clustering
        :return: int, number of communities
        """

        comid = self.communities()
        return len(np.unique(comid[np.isfinite(comid)]))

    def clusterorder(self, secondary=None, tertiary=None):
        """
        Order cells for display by cluster, first, then by a secondary vector
        :param secondary: a secondary vector. If blank, sorted based on node index. e.g. cell type
        :param tertiary: optional tertiary sort, e.g. cell response magnitude
        :return: sorted numpy array of indices
        """

        primary = self.communities()[self.limits]
        if secondary is None:
            secondary = np.ones(np.sum(self.limits))
        elif len(secondary) == self.ncells and len(primary) < len(secondary):
            secondary = np.array(secondary)[self.limits]
        if tertiary is None:
            tertiary = np.ones(len(primary))
        elif len(tertiary) == self.ncells and len(primary) < len(tertiary):
            tertiary = np.array(tertiary)[self.limits]

        comb = np.concatenate([sorted(self.nodes), primary, secondary, tertiary]).reshape(4, len(primary))
        inds = np.lexsort((comb[0, :], comb[3, :], comb[2, :], comb[1, :]))
        return self.nodes[inds]

    def label(self, groups):
        """
        Add labels to each of the nodes
        :param groups: list of string labels or numpy boolean
        :return: None
        """

        self.labels = groups

    def clusterlabel(self, label='', exclusive=False, count=1, labels=None):
        """
        Group into those cells that cluster with label label. Remove labeled cells if exclusive.
        :param label: label string to use if labeled groups was strings
        :param exclusive: exclude labeled cells if true
        :param count: the minimum number of labeled cells per cluster
        :param labels: add labels beyond using the label function
        :return: two boolean arrays of clustered-with-label, and excluded-from-label
        """

        # Check input types
        if labels is not None:
            self.labels = labels

        if isinstance(self.labels, list) and isinstance(self.labels[0], str) and len(label) > 0:
            keys = np.zeros(len(self.labels)) > 1
            for i in range(len(self.labels)):
                if self.labels[i] == label:
                    keys[i] = True
        elif isinstance(self.labels, np.ndarray):
            keys = self.labels
        else:
            raise LookupError('Labels used incorrectly')

        # Count labels per cluster
        ncoms = self.ncommunities()
        comlabels = self.communities()
        labelcounts = {com: 0 for com in range(ncoms)}
        for node in self.nodes:
            if keys[node]:
                labelcounts[comlabels[node]] += 1

        # Assign TF based on labels per cluster
        incl = np.zeros(self.ncells) > 1
        for node in self.nodes:
            if comlabels[node] in labelcounts and labelcounts[comlabels[node]] >= count:
                incl[node] = True

        # Clean up and calculate inverse
        if self.limits is None:
            excl = np.invert(incl)
        else:
            excl = np.bitwise_and(self.limits, np.invert(incl))

        if exclusive:
            incl[keys] = False

        return incl, excl

    def groupconnectivity(self):
        """
        Measure the connectivity within each group relative to outside each group
        :return: tuple of vector within group, vector out of group
        """

        comid = self.communities()
        ncomms = len(np.unique(comid[np.isfinite(comid)]))

        ingraph, outgraph = self.gx.copy(), self.gx.copy()
        for c1, c2, weight in self.edges:
            if comid[c1] == comid[c2]:
                outgraph.remove_edge(c1, c2)
            else:
                ingraph.remove_edge(c1, c2)

        iconn, oconn = [], []

        try:
            iclust, oclust = nx.clustering(ingraph, None, weight='weight'), nx.clustering(outgraph, None, weight='weight')
            for c1 in self.nodes:
                iconn.append(iclust[c1])
                oconn.append(oclust[c1])
        except ZeroDivisionError:
            pass

        return iconn, oconn

    def labelconnectivity(self, label, exclusive=False, count=1, labels=None):
        """
        Measure the connectivity within each group, split by labeled and unlabeled, and between groups of different
        labels, split by labeled and unlabeled
        :param label: label string to use if labeled groups was strings
        :param exclusive: exclude labeled cells if true
        :param count: the minimum number of labeled cells per cluster
        :param labels: add labels beyond using the label function
        :return: tuple of vector within group, vector out of group
        """

        comid = self.communities()
        ncomms = len(np.unique(comid[np.isfinite(comid)]))
        incl, excl = self.clusterlabel(label, exclusive, count, labels)

        ingraph, outgraph = self.gx.copy(), self.gx.copy()
        for c1, c2, weight in self.edges:
            if comid[c1] != comid[c2]:
                ingraph.remove_edge(c1, c2)

            if comid[c1] == comid[c2] or (incl[c1] and not incl[c2]) or (excl[c1] and not excl[c2]):
                outgraph.remove_edge(c1, c2)

        iconninc, oconninc, iconnexc, oconnexc = [], [], [], []

        try:
            iclust, oclust = nx.clustering(ingraph, None, weight='weight'), nx.clustering(outgraph, None, weight='weight')
            for c1 in self.nodes:
                if incl[c1]:
                    iconninc.append(iclust[c1])
                    oconninc.append(oclust[c1])

                if excl[c1]:
                    iconnexc.append(iclust[c1])
                    oconnexc.append(oclust[c1])

        except ZeroDivisionError:
            pass

        # Within included (matches label), between included, within excluded, between excluded
        return iconninc, oconninc, iconnexc, oconnexc

    def relativeconnectivity(self, ctype='correlations', label='', exclusive=False, count=1):
        """
        Compare the within-group vs out-group connectivity levels
        :param ctype: connectivity type, can be 'clustering' or 'correlations'
        :param label: optional label string to use if labeled groups was strings. This will return a dict
        :param exclusive: exclude labeled cells if true
        :param count: the minimum number of labeled cells per cluster
        """

        edgemat = np.zeros((len(self.nodes), len(self.nodes)))
        for c1, c2, weight in self.edges:
            edgemat[c1, c2] = weight

        # Return only ingroup, outgroup
        if len(label) == 0:
            gin = []
            gout = []

            #for node in self.nodes:


        # Return ingroup-special, ingroup, outgroup
        else:
            pass

def nodes_edges(nodes, corr, limits=None, addzeros=False):
    """
    Get nodes and edges given a type of noise correlation and a limitation such as visual-drivenness
    :param nodes: can be int or a list defining the range
    :param corr: correlations that define edges
    :param limits: limitations on which nodes to include
    :return: nodes numpy array, edges list
    """

    # Check types
    if isinstance(nodes, int):
        nodes = np.arange(nodes)

    nodes = np.array(nodes)
    if limits is not None:
        nodes = nodes[limits]

    # We're setting negative correlations to 0 for community analysis reasons
    corr[np.isnan(corr)] = 0
    corr[corr < 0] = 0

    # Format all edge weights for a networkx graph
    mxweight = np.nanmax(corr)
    edges = []
    if mxweight > 0:
        for i, c1 in enumerate(nodes):
            for c2 in nodes[i + 1:]:
                if corr[c1, c2] != np.nan and corr[c1, c2] > 0:
                    edges.append((c1, c2, corr[c1, c2]))
                elif addzeros:
                    edges.append((c1, c2, 0))

    return nodes, edges

def nxgraph(nodes, corr, limits=None):
    """
    Make a new NCGraph instance for a NetworkX graph.
    Do not set with nodes as a list or it will return something
    that cannot be compared with the cells in a particular day.

    Parameters
    ----------
    nodes : int
        Number of cells in the day
    corr : matrix of ncells x ncells
        The noise correlations or other correlations from which edge weights should be set
    limits : boolean or index array
        Limiting which nodes should be included, such as visual-drivenness

    Returns
    -------

    """

    gx = NCGraph(nodes, corr, limits)
    return gx
