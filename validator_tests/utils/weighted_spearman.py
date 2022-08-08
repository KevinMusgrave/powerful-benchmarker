import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from WeightedCorr import WeightedCorr


def set_nan_inf_to_min(x):
    x = x.copy()
    is_finite = np.isfinite(x)
    x[~is_finite] = np.min(x[is_finite])
    return x


def assert_all_finite(target_accuracies, validation_scores):
    validation_scores = set_nan_inf_to_min(validation_scores)
    assert np.isfinite(target_accuracies).all()
    assert np.isfinite(validation_scores).all()
    return target_accuracies, validation_scores


def weighted_spearman(target_accuracies, validation_scores, pow):
    target_accuracies, validation_scores = assert_all_finite(
        target_accuracies, validation_scores
    )
    ranks = rankdata(validation_scores, method="dense").astype(float)
    ranks /= np.max(ranks)
    weights = ranks**pow

    return WeightedCorr(
        x=pd.Series(validation_scores),
        y=pd.Series(target_accuracies),
        w=pd.Series(weights),
    )(method="spearman")


def spearman(target_accuracies, validation_scores):
    target_accuracies, validation_scores = assert_all_finite(
        target_accuracies, validation_scores
    )
    return spearmanr(
        target_accuracies,
        validation_scores,
    ).correlation
