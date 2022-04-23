def Entropy():
    flags = []
    for split in ["src_train", "src_val", "target_train"]:
        flags.append({"validator": "Entropy", "split": split})
    return flags
