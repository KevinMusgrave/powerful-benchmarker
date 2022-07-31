import numpy as np
from scipy.stats import rankdata


def special_min_max_normalize(x):
    max_min = max(x) - min(x)
    if max_min == 0:
        max_x = max(x)
        if max_x == 0:
            return x + 1
        return x / max_x
    return (x - min(x)) / max_min


def tie_break_ndcg(scores, ranks, pow):
    bincounts = np.bincount(ranks)
    scales = 1.0 / bincounts[ranks]
    return np.sum((scores * scales) / (ranks**pow))


# tied-rank normalized discounted cumulative gain
def trndcg_score(target_accuracies, validation_scores, pow):
    assert len(target_accuracies) == len(validation_scores)
    target_accuracies = special_min_max_normalize(target_accuracies)
    sort_idx = np.argsort(validation_scores)[::-1]
    y = target_accuracies[sort_idx]
    v_ranks = rankdata(-validation_scores[sort_idx], method="dense")
    dcg = tie_break_ndcg(y, v_ranks, pow)
    sorted_target_accuracies = np.sort(target_accuracies)[::-1]
    true_ranks = rankdata(-sorted_target_accuracies, method="dense")
    best_dcg = tie_break_ndcg(sorted_target_accuracies, true_ranks, pow)
    score = dcg / best_dcg
    return score if score < 1 else 1 / score
