import numpy as np

def pinball_loss(y_true, y_pred, q):
    error = y_true - y_pred
    return np.mean(np.maximum(q * error, (q - 1) * error))


def quantile_coverage(y_true, y_pred):
    return np.mean(y_true <= y_pred)
