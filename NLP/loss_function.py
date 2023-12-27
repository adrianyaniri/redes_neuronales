import numpy as np

def mean_squared_error_loss(predicted, actual):
    return np.mean((predicted - actual) ** 2)