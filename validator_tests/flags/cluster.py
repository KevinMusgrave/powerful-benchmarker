def DomainCluster():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for normalize in [0, 1]:
            for p in [2]:
                flags.append(
                    {
                        "validator": "DomainCluster",
                        "split": "train",
                        "layer": layer,
                        "normalize": str(normalize),
                        "p": str(p),
                    }
                )
    return flags


def ClassAMI():
    flags = []
    for layer in ["features", "logits"]:
        for normalize in [0, 1]:
            for p in [2]:
                for with_src in [0, 1]:
                    flags.append(
                        {
                            "validator": "ClassAMI",
                            "split": "train",
                            "layer": layer,
                            "normalize": str(normalize),
                            "p": str(p),
                            "with_src": str(with_src),
                        }
                    )
    return flags


def ClassAMICentroidInit():
    flags = ClassAMI()
    for f in flags:
        f["validator"] = "ClassAMICentroidInit"
    return flags


def ClassSS():
    flags = ClassAMI()
    for f in flags:
        f["validator"] = "ClassSS"
    return flags


def ClassSSCentroidInit():
    flags = ClassAMI()
    for f in flags:
        f["validator"] = "ClassSSCentroidInit"
    return flags
