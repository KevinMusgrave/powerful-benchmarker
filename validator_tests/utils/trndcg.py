import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .weighted_corr import WeightedCorr


def set_nan_inf_to_min(x):
    is_finite = np.isfinite(x)
    x[~is_finite] = np.min(x[is_finite])


def special_min_max_normalize(x):
    set_nan_inf_to_min(x)
    max_min = max(x) - min(x)
    if max_min == 0:
        max_x = max(x)
        if max_x == 0:
            return x + 1
        return x / max_x
    return (x - min(x)) / max_min


def weighted_spearman(target_accuracies, validation_scores, pow):
    set_nan_inf_to_min(validation_scores)
    assert np.isfinite(target_accuracies).all()
    assert np.isfinite(validation_scores).all()

    ranks = rankdata(validation_scores, method="dense").astype(float)
    ranks /= np.max(ranks)
    weights = ranks**pow

    return WeightedCorr(
        x=pd.Series(validation_scores),
        y=pd.Series(target_accuracies),
        w=pd.Series(weights),
    )(method="spearman")
