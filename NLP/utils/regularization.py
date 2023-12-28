import numpy as np


def calculate_regularization(weights):
    regularization_coefficient = 0.0001
    return regularization_coefficient * sum(np.square(w) for w in weights[:-1])

