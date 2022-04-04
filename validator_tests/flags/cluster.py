def DomainCluster():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for normalize in [0, 1]:
            flags.append(
                {
                    "validator": "DomainCluster",
                    "split": "train",
                    "layer": layer,
                    "normalize": str(normalize),
                }
            )
    return flags


def ClassCluster():
    flags = DomainCluster()
    for f in flags:
        f["validator"] = "ClassCluster"
    return flags
