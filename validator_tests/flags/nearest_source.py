def NearestSource():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for threshold in [0, 0.5]:
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
