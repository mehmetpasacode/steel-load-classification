import numpy as np


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(targets, probabilities, class_weights=None):
    per_row = -np.sum(targets * np.log(np.clip(probabilities, 1e-12, 1)), axis=1)
    if class_weights is not None:
        per_row *= targets @ class_weights
    return per_row.mean()

