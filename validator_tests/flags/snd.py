def SND():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for T in [0.01, 0.05, 0.1, 0.5]:
            flags.append(
                {
                    "validator": "SND",
                    "split": "target_train",
                    "layer": layer,
                    "T": str(T),
                }
            )
    return flags
