def Accuracy():
    flags = []
    for average in ["micro", "macro"]:
        for split in ["src_train", "src_val", "target_train", "target_val"]:
            flags.append(f"--validator Accuracy --average={average} --split={split}")
    return flags
