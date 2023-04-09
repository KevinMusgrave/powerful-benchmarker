import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from WeightedCorr import WeightedCorr


def set_nan_inf_to_min(x):
    x = x.copy()
    is_finite = np.isfinite(x)
    x[~is_finite] = np.min(x[is_finite])
    return x


def assert_all_finite(validation_scores, target_accuracies):
    validation_scores = set_nan_inf_to_min(validation_scores)
    assert np.isfinite(target_accuracies).all()
    assert np.isfinite(validation_scores).all()
    return validation_scores, target_accuracies


def weighted_spearman(validation_scores, target_accuracies, pow):
    validation_scores, target_accuracies = assert_all_finite(
        validation_scores, target_accuracies
    )
    v_ranks = rankdata(validation_scores, method="dense").astype(float)
    v_ranks /= np.max(v_ranks)
    t_ranks = rankdata(target_accuracies, method="dense").astype(float)
    t_ranks /= np.max(t_ranks)
    weights = np.maximum(v_ranks, t_ranks) ** pow

    return WeightedCorr(
        x=pd.Series(validation_scores),
        y=pd.Series(target_accuracies),
        w=pd.Series(weights),
    )(method="spearman")


def spearman(validation_scores, target_accuracies):
    validation_scores, target_accuracies = assert_all_finite(
        validation_scores, target_accuracies
    )
    return spearmanr(validation_scores, target_accuracies).correlation
