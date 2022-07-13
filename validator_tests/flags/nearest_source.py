def NearestSource():
    flags = []
    for layer in ["logits"]:
        for threshold in [-2]:
            for weighted in [0, 1]:
                flags.append(
                    {
                        "validator": "NearestSource",
                        "layer": layer,
                        "threshold": threshold,
                        "weighted": weighted,
                    }
                )
    return flags


def NearestSourceL2():
    flags = []
    for layer in ["features", "logits", "preds"]:
        flags.append(
            {
                "validator": "NearestSourceL2",
                "layer": layer,
            }
        )
    return flags
