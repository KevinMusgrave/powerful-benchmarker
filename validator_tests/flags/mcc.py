def MCC():
    flags = []
    for split in ["src_train", "target_train", "src_val"]:
        for T in [0.1, 1, 10]:
            flags.append({"validator": "MCC", "split": split, "T": str(T)})
    return flags
