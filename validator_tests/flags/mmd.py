def MMD():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for exponent in [0, 4, 8]:
            flags.append(
                {
                    "validator": "MMD",
                    "split": "train",
                    "layer": layer,
                    "exponent": str(exponent),
                }
            )
    return flags
