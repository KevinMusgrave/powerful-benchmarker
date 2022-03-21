def KNN():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for k in [1, 10]:
            for normalize in [0, 1]:
                # no point in trying l2 normalized preds because
                # they are already normalized by softmax
                if layer == "preds" and normalize == 1:
                    continue
                flags.append(
                    {
                        "validator": "KNN",
                        "split": "train",
                        "layer": layer,
                        "k": str(k),
                        "normalize": str(normalize),
                    }
                )
    return flags
