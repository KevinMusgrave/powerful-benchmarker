def DEV():
    return [
        f"--validator DEV --layer={layer}" for layer in ["features", "logits", "preds"]
    ]
