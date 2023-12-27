import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from functionActivacion import sigmoid, sigmoid_derivada
from loss_function import mean_squared_error_loss
from generar_datos import generate_nonlinear_dataset

class NLP:
    
    def __init__(self, num_layers, num_neur_layers, function, loss_function):
        self.num_layers = num_layers
        self.num_neur_layers = num_neur_layers
        self.function = function
        self.loss_function = loss_function
        
        self.activations = []
        self.errors = []
        self.learning_rate = 0.01
        
        # Inicializacion de los pesos
        self.weights = [np.random.rand(num_neur_layers[i], num_neur_layers[i + 1]) for i in range(num_layers - 1)]
            
        # Inicializar los bias de la red
        self.biases = [np.random.rand(num_neur_layers[i + 1]) for i in range(num_layers - 1)]


    def forward(self, x):
        self.activations = [x]
        for i in range (self.num_layers - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            act = self.function(z)
            self.activations.append(act)
        return self.activations[-1]       # Activacion de la ultima capa
        
    
    def activation(self, x):
        # Aplica la funcion de activacion
        return self.function(x)
    
    def loss(self,prediction, actual):
        return self.loss_function(prediction, actual)
        
    
    def backpropagation(self, x, y_true):
        # Calcula la salida 
        y_pred = self.forward(x)
    
        # Calcula el error
        error = self.loss(y_pred, y_true)

        # Inicializa self.errors antes de calcular las derivadas
        self.errors = [error]

        # Calcular las derivadas
        # Revisar la g(h)
        d_error_activations = [sigmoid_derivada(a) * error for a in self.activations]
        d_error_weights = [np.dot(self.activations[i].T, d_error_activations[i + 1]) for i in range(self.num_layers - 1)]
        d_error_biases = [np.sum(d_error_activations[i + 1], axis=0) for i in range(self.num_layers - 1)]

        # Actualizar los pesos y bias
        self.weights = [w - self.learning_rate * dw for w, dw in zip(self.weights, d_error_weights)]
        self.biases = [b - self.learning_rate * db for b, db in zip(self.biases, d_error_biases)]
    
        return error
    
    def train(self, X_train, y_train, epochs, learning_rate):
        self.learning_rate = learning_rate

        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X_train, y_train):
                # Asegúrate de llamar a forward antes de backpropagation
                self.forward(x)
                error = self.backpropagation(x, y)
                total_loss += error

            loss = total_loss / len(X_train)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss}')


# Inicializacion de las capas y neuronas
input_size = 3
hidden_size = 5
output_size = 1

# Instancia del NLP
nlp_instance = NLP(num_layers=3, num_neur_layers=[input_size, hidden_size, output_size], function=sigmoid, loss_function=mean_squared_error_loss)

# Generar conjunto de datos
X_train, y_train = generate_nonlinear_dataset()

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Entrenamiento de la red
epochs = 5000
learning_rate = 0.1
for epoch in range(epochs):
    total_loss = 0
    for x, y in zip(X_train, y_train):
        x = np.expand_dims(x, axis=0)  # Asegurar que x tenga forma (1, 2)
        y = np.array([y])  # Asegurar que y tenga forma (1,)

        # Asegúrate de llamar a forward antes de backpropagation
        nlp_instance.forward(x)
        error = nlp_instance.backpropagation(x, y)
        total_loss += error

    average_loss = total_loss / len(X_train)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

# Realizar predicciones en el conjunto de datos de entrenamiento
predictions = np.array([nlp_instance.forward(np.expand_dims(x, axis=0))[0] for x in X_train])


class_0_indices = np.where(predictions < 0.5)[0]
class_1_indices = np.where(predictions >= 0.5)[0]

# Visualizar las predicciones y el conjunto de datos
plt.scatter(X_train[class_0_indices, 0], X_train[class_0_indices, 1], label='Clase 0 (Predicción)', marker='o', alpha=0.5)
plt.scatter(X_train[class_1_indices, 0], X_train[class_1_indices, 1], label='Clase 1 (Predicción)', marker='x', alpha=0.5)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='Clase 0 (Real)', marker='o')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='Clase 1 (Real)', marker='x')

plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.legend()
plt.show()

print('Predicciones en el conjunto de prueba:')
print(predictions)


# Prediciones
predictions_test = np.array([nlp_instance.forward(np.expand_dims(x, axis=0))[0] for x in X_test])
# Convertir las predicciones a etiquetas binarias (0 o 1) usando un umbral
binary_predictions = (predictions_test >= 0.5).astype(int)

# Calcular la precisión del modelo en el conjunto de prueba
accuracy = accuracy_score(y_test, binary_predictions)
print(f'Precisión en el conjunto de prueba: {accuracy * 100:.2f}%')