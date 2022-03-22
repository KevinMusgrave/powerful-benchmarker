def AMI():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for normalize in [0, 1]:
            flags.append(
                {
                    "validator": "AMI",
                    "split": "train",
                    "layer": layer,
                    "normalize": str(normalize),
                }
            )
    return flags
