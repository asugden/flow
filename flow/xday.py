import numpy as np
import os.path as opath

from . import paths


def ids(mouse, day1, day2):
    """Return those IDs that are found on both days with mutual information greater than the threshold

    :param mouse: mouse name, str
    :param day1: date, str, yymmdd
    :param day2: date, str
    :return: positions that match on both days

    """
    day1, day2 = str(day1), str(day2)
    ids1, ids2 = paths.pairids(mouse, day1, day2)

    if len(ids1) > 0 or len(ids2) > 0:
        return np.array(ids1), np.array(ids2)

    else:
        ids1 = np.array([int(id) for id in _read_crossday_ids(mouse, day1)])
        ids2 = np.array([int(id) for id in _read_crossday_ids(mouse, day2)])
        if len(ids1) == 0 or len(ids2) == 0:
            return [], []

        matchpos1 = [i for i, id in enumerate(ids1) if id > 0 and id in ids2]
        matchids1 = [id for i, id in enumerate(ids1) if id > 0 and id in ids2]
        matchpos2 = [np.argmax(ids2 == id) for id in matchids1]

        return np.array(matchpos1), np.array(matchpos2)

def _read_crossday_ids(mouse, date):
    """
    Read ID file.
    """

    idf = paths.ids(mouse, str(date))
    if len(idf) == 0 or not opath.exists(idf):
        return []

    out = []
    if len(idf) > 0:
        fp = open(idf, 'r')
        allids = fp.read()
        fp.close()

        # Analyze each ID
        allids = allids.split('\n')
        if allids[-1] == '':
            allids = allids[:-1]

        # Convert to ints
        out = allids

    return out
