def BNM():
    flags = []
    for split in ["src_train", "target_train", "src_val"]:
        flags.append({"validator": "BNM", "layer": "logits", "split": split})
    return flags


def BSP():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for split in ["src_train", "target_train"]:
            for k in [1]:
                flags.append(
                    {"validator": "BSP", "layer": layer, "split": split, "k": str(k)}
                )
    return flags


def FBNM():
    flags = BNM()
    for f in flags:
        f["validator"] = "FBNM"
    return flags
