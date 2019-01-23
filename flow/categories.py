from builtins import range
import numpy as np

from . import glm

def labels(day, categories=('lick', 'ensure', 'reward', 'non'), cluster_numbers=None, additional=None):
    """
    Return single labels for cells based on a GLM, optional clustering,
    and optional additional categories.

    Parameters
    ----------
    day : Date instance
    categories : tuple of strings
        An ordered list by which labels will be added. A cell will be
        assigned to the first category to which it matches or undefined.
    cluster_numbers : array of ints
        The array number associated with any cell, used for reward clustering.
    additional : dict of arrays of ints
        For labels not derived from the GLM such as visually-driven, these
        can be added in another dictionary which will be combined with the glm
        dictionary.

    Returns
    -------
    list of strings
        A string label for each cell

    """

    # Get GLM labels
    lbls = glm.glm(day.mouse, day.date).labels()
    if additional is not None:
        for key in additional:
            lbls[key] = additional[key]
    ncells = len(lbls[list(lbls.keys())[0]])

    # Add reward and non, if desired
    if cluster_numbers is not None:
        lbls['reward'] = np.zeros(ncells) > 1
        lbls['non'] = np.zeros(ncells) > 1
        clustered = np.where(np.isfinite(cluster_numbers))[0]
        rewards = {com: 0 for com in np.unique(cluster_numbers[np.isfinite(cluster_numbers)])}

        for node in clustered:
            if lbls['ensure'][node]:
                rewards[cluster_numbers[node]] += 1

        for node in clustered:
            if rewards[cluster_numbers[node]] > 0:
                lbls['reward'][node] = True
            else:
                lbls['non'][node] = True

    # And convert to single categories
    out = ['undefined' for i in range(ncells)]
    for c in range(ncells):
        for cat1 in categories:
            if lbls[cat1][c]:
                out[c] = cat1
                break

    return out
