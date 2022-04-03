def KNN():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for k in [1000]:
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


def TargetKNN():
    flags = []
    for k in [1000]:
        for p in [2]:
            for normalize in [0, 1]:
                for T_in_ref in [0, 1]:
                    flags.append(
                        {
                            "validator": "TargetKNN",
                            "split": "train",
                            "k": str(k),
                            "p": str(p),
                            "normalize": str(normalize),
                            "T_in_ref": str(T_in_ref),
                        }
                    )
    return flags


def TargetKNNLogits():
    flags = TargetKNN()
    for f in flags:
        f["validator"] = "TargetKNNLogits"
    return flags
