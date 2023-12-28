import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from functionActivacion import sigmoid, sigmoid_derivada
from loss_function import mean_squared_error_loss
from generar_datos import generate_nonlinear_dataset
from utils.regularization import calculate_regularization


class NLP:

    def __init__(self, num_layers_total, num_neurons_layers, activation_function, errors_function):
        """
        Inicializacion de los atributos de la clase NLP

        :param num_layers_total: Numeros de capas de la red
        :param num_neurons_layers: Numberos de neuronas de la capa
        :param activation_function: Function de activacion
        :param errors_function: Funcion de error de la capa
        """

        self.num_layers_total = num_layers_total
        self.num_neurons_layers = num_neurons_layers
        self.activation_function = activation_function
        self.errors_function = errors_function

        self.errors = []

        self.activations = []
        self.learning_rate = 0.01

        # Inicializar de los pesos y bias
        self.weights = []
        self.biases = []

        for i in range(num_layers_total - 1):
            weights = np.random.randn(num_neurons_layers[i], num_neurons_layers[i + 1]) * 0.01
            biases = np.random.randn(num_neurons_layers[i + 1]) * 0.01
            self.weights.append(weights)
            self.biases.append(biases)

        # Regularizacion L2
        self.regularization_coefficient = 0.0001

    def forward(self, x):
        """
        Realiza la propagacion hace adelante de la red
        :param x: Entrada de la red
        :return: Activacion de la ultima capa
        matmul -> calcula la suma ponderada
        """
        # capa entrada
        activations = x
        for i in range(self.num_layers_total - 1):
            # Capas oculta
            weights = self.weights[i]
            biases = self.biases[i]
            activations = self.activation_function(np.matmul(activations, weights) + biases)
            self.activations.append(activations)

        return activations

    def activation(self, x):
        """
        Aplica la funcion de activaciones
        :param x: Valor al aplicar la funcion de activacion
        :return: Resultado de aplicar la activacion
        """

        return self.activation_function(x)

    def loss(self, prediction, actual):
        """
        Calcula la funcion de perdida
        :param prediction: Prediccion de la red
        :param actual: Valor real
        :return: Valor de la funcion de perdida
        """
        return self.errors_function(prediction, actual)

    def backpropagation(self, x, y_true):
        """
        Realiza la propagacion hacia atras
        :param x:
        :param y_true:
        :return:
        """
        # Calcula la salida
        y_pred = self.forward(x)

        # Calcula el error
        error = self.errors_function(y_pred, y_true)

        # Inicializa self.errors antes de calcular las derivadas
        self.errors = [error]
        self.activations = [x]

        # Inicializa delta_activations con la forma adecuada para la primera capa
        delta_activations = np.zeros((self.num_layers_total - 1, self.num_neurons_layers[0]))
        # Calculas los deltas de las activaciones para cada neurona
        delta_activations = sigmoid_derivada(y_pred) * error

        # Calcula la regularizacion
        regularization = calculate_regularization(self.weights)

        print("Longitud de self.activations antes del bucle:", len(self.activations))
        for i in reversed(range(self.num_layers_total - 1)):
            # Actualización de delta_activations para la siguiente iteración:
            if i > 0:  # No actualizar en la última capa
                print("Dimensiones de delta_activations[i-1]: ", delta_activations[i - 1].shape)
                delta_activations = np.matmul(self.weights[i].T,
                                              delta_activations[i - 1]) * sigmoid_derivada(self.activations[i - 1])
            print("Dimensiones de self.weights[i].T: ", self.weights[i].T.shape)
            delta_weights = np.matmul(self.weights[i].T,
                                      delta_activations[i - 1]) * sigmoid_derivada(self.activations[i - 1])
            delta_biases = np.sum(delta_activations[i], axis=0)

            self.weights[i - 1] = (self.weights[i - 1] - (self.learning_rate * delta_weights)
                                   - (self.learning_rate * regularization))

            self.biases[i - 1] = self.biases[i - 1] - self.learning_rate * delta_biases
            self.activations.append(self.activations[i])

        return {'weights': self.weights, 'biases': self.biases}

    def train(self, X_train, y_train, epochs, batch_size):
        """
        Funcion de entranamiento para la red
        :param batch_size:
        :param X_train: Conjunto de entrenamiento
        :param y_train: Etiquetas del conjunto de entrenamiento
        :param epochs: Numero de epocas de entrenamiento
        """
        for epoch in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)  # Para asegurar de que el conjunto no este ordenado
            for i in range(0, len(X_train), batch_size):
                x_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                self.backpropagation(x_batch, y_batch)

    def predict(self, x):
        return self.forward(x)

    def evaluate(self, X_test, y_test, loss_function=None):
        """
        Calcula el error de validacion
        :param loss_function:
        :param X_test: Conjunto de prueba
        :param y_test: Etiquetas del conjunto de prueba
        :return: Error de validacion
        """
        if loss_function is None:
            loss_function = self.loss()
        # Realiza la propagacion hacia adelante
        outputs = self.forward(X_test)

        # Calcula el error
        error = self.loss(X_test, y_test)
        return error


# Inicializacion de las capas y neuronas
input_size = 3
hidden_size = 5
output_size = 1

epochs = 50
learning_rate = 0.01
batch_size = 32

# Generar conjunto de datos
X_train, y_train = generate_nonlinear_dataset()

# Separacion de los conjunto de train y test
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalizacion de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Instancia del NLP
nlp = NLP(
    num_layers_total=3,
    num_neurons_layers=[input_size, hidden_size, output_size],
    activation_function=sigmoid,
    errors_function=mean_squared_error_loss
)

# Entrenamiento de la red
nlp.train(X_train, y_train, epochs, batch_size)

# Realizar predicciones en el conjunto de datos de entrenamiento
predictions = np.array([nlp.forward(np.expand_dims(x, axis=0))[0] for x in X_train])

# Visualizacion

pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

plt.scatter(X_train_reduced[y_train == 0, 0], X_train_reduced[y_train == 0, 1], label='Clase 0 (Real)', marker='o',
            alpha=0.5)
plt.scatter(X_train_reduced[y_train == 1, 0], X_train_reduced[y_train == 1, 1], label='Clase 1 (Real)', marker='x',
            alpha=0.5)
plt.scatter(X_train_reduced[predictions < 0.5, 0], X_train_reduced[predictions < 0.5, 1], label='Clase 0 (Predicción)',
            marker='o', alpha=0.5)
plt.scatter(X_train_reduced[predictions >= 0.5, 0], X_train_reduced[predictions >= 0.5, 1],
            label='Clase 1 (Predicción)', marker='x', alpha=0.5)
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.show()

# Evaluación del modelo

# Predicciones en el conjunto de prueba
predictions_test = np.array([nlp.forward(np.expand_dims(x, axis=0))[0] for x in X_test])

# Convertir las predicciones a etiquetas binarias (0 o 1) usando un umbral
binary_predictions = (predictions_test >= 0.5).astype(int)

# Calcular la precisión del modelo en el conjunto de prueba
accuracy = accuracy_score(y_test, binary_predictions)
print(f'Precisión en el conjunto de prueba: {accuracy * 100:.2f}%')

# Otros métricas de evaluación

precision = precision_score(y_test, binary_predictions)
recall = recall_score(y_test, binary_predictions)
f1 = f1_score(y_test, binary_predictions)
auc = roc_auc_score(y_test, predictions)

print(f'Precisión: {precision}')
print(f'Sensibilidad: {recall}')
print(f'F1-score: {f1}')
print(f'AUC: {auc}')
