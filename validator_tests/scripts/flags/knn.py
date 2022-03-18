def KNN():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for k in [1, 10]:
            for l2_normalize in [0, 1]:
                # no point in trying l2 normalized preds because
                # they are already normalized by softmax
                if layer == "preds" and l2_normalize == 1:
                    continue
                flags.append(
                    f"--validator KNN --split=train --layer={layer} --k={k} --l2_normalize={l2_normalize}"
                )
    return flags
