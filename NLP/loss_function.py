import numpy as np


def mean_squared_error_loss(predicted, actual):
    # Ajusta las formas de los arrays si es necesario
    predicted = np.squeeze(predicted)  # Elimina dimensiones adicionales
    actual = np.reshape(actual, predicted.shape)  # Redimensiona actual

    return np.mean((predicted - actual) ** 2)
