def SilhouetteScore():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for with_src in [0, 1]:
            for pca_size in [64]:
                flags.append(
                    {
                        "validator": "SilhouetteScore",
                        "layer": layer,
                        "with_src": str(with_src),
                        "pca_size": str(pca_size),
                    }
                )
    return flags


def CHScore():
    flags = SilhouetteScore()
    for f in flags:
        f["validator"] = "CHScore"
    return flags
