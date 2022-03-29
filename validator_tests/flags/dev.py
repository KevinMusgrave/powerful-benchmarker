def DEV():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for min_var in [0, 0.01]:
            flags.append({"validator": "DEV", "layer": layer, "min_var": str(min_var)})
    return flags
