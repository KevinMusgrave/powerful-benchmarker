def Diversity():
    flags = []
    for split in ["src_train", "src_val", "target_train"]:
        flags.append({"validator": "Diversity", "split": split})
    return flags
