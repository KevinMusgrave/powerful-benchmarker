def IST():
    flags = []
    for layer in ["features", "logits", "preds"]:
        for with_ent, with_div in [(0, 1), (1, 0)]:
            flags.append(
                f"--validator IST --split=train --layer={layer} --with_ent={with_ent} --with_div={with_div}"
            )
    return flags
