def DEV():
    return [
        {"validator": "DEV", "layer": layer}
        for layer in ["features", "logits", "preds"]
    ]
