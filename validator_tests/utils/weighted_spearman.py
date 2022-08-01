import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .weighted_corr import WeightedCorr


def set_nan_inf_to_min(x):
    is_finite = np.isfinite(x)
    x[~is_finite] = np.min(x[is_finite])
    return x


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


def spearman(target_accuracies, validation_scores):
    return spearmanr(
        target_accuracies,
        set_nan_inf_to_min(validation_scores),
    ).correlation
