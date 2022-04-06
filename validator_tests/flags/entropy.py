def Entropy():
    flags = []
    for split in ["src_train", "target_train"]:
        flags.append({"validator": "Entropy", "split": split})
    return flags
