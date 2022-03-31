def DEV():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for normalization in ["None", "max", "standardize"]:
            flags.append(
                {"validator": "DEV", "layer": layer, "normalization": normalization}
            )
    return flags
