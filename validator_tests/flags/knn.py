def KNN():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for k in [1, 10]:
            for p in [2]:
                for normalize in [0, 1]:
                    flags.append(
                        {
                            "validator": "KNN",
                            "split": "train",
                            "layer": layer,
                            "k": str(k),
                            "p": str(p),
                            "normalize": str(normalize),
                        }
                    )
    return flags
