import numpy as np


def nannone(val):
    """Convert any Nones to NaN."""

    if val is None:
        return np.nan
    else:
        return val


def emptynone(val):
    """Convert any Nones to empty lists."""

    if val is None:
        return []
    else:
        return val
