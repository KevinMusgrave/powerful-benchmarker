def IST():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for with_ent, with_div in [(0, 1), (1, 0)]:
            flags.append(
                {
                    "validator": "IST",
                    "split": "train",
                    "layer": layer,
                    "with_ent": str(with_ent),
                    "with_div": str(with_div),
                }
            )
    return flags
