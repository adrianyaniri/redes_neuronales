# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def generate_nonlinear_dataset(num_samples=5):
    np.random.seed(42)

    # Genera puntos aleatorios en el rango [-1, 1] para las dos características
    X = np.random.uniform(low=-1, high=1, size=(num_samples, 3))

    # Etiqueta de clase 1 para puntos en el cuadrante superior derecho y cuadrante inferior izquierdo
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    X[:, 2] = 1

    return X, y


# Visualización del conjunto de datos
X, y = generate_nonlinear_dataset()
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Clase 0', marker='o')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Clase 1', marker='x')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.show()
