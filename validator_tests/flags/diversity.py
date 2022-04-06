def Diversity():
    flags = []
    for split in ["src_train", "target_train"]:
        flags.append({"validator": "Diversity", "split": split})
    return flags
