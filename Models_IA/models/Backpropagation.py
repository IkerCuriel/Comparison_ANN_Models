import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class Backpropagation_Low_Level:
    def __init__(self, x, y, l_r, alpha, epochs):
        # Parámetros
        self.x = x
        self.y = y 
        self.l_r = l_r
        self.alpha = alpha  # momentum
        self.epochs = epochs
        
        # Inicialización de los pesos y los sesgos
        np.random.seed(0)
        self.weights = 2 * np.random.random((3, 1)) - 1
        self.bias = 2 * np.random.random((1, 1)) - 1
        self.w_new = 0
        self.b_new = 0
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
        
    def train(self, x):
        # Entrenamiento con momentum
        errors_with_momentum = []
        for i in range(self.epochs):
            # Feedforward
            z = np.dot(self.x, self.weights) + self.bias
            yo = self.sigmoid(z)

            # Backpropagation
            error = self.y - yo
            derived_e = error * self.sigmoid_derivative(yo)

            # Actualizar pesos y sesgos
            delta_w = np.dot(self.x.T, derived_e) #Transpuesta
            delta_b = np.sum(derived_e, axis=0, keepdims=True)

            # Momentum
            weights += self.l_r * delta_w + self.alpha * (self.w_new - self.weights)
            bias += self.l_r * delta_b + self.alpha * self.b_new
            
            # Guardar error cuadrático medio
            errors_with_momentum.append(np.mean(error**2)) 
            
        # Imprimir la salida deseada y la salida calculada
        print("Salida deseada (y):")
        print(self.y)
        print("\nSalida calculada (yo):")
        print(yo) 

def Backpropagation_High_Level(x, y, epochs):
    # Definición del modelo en Keras
    model = Sequential()
    model.add(Dense(1, input_dim=3, activation='sigmoid'))

    # Compilación del modelo
    model.compile(loss='mean_squared_error', optimizer='sgd')

    # Entrenamiento del modelo
    model.fit(x, y, epochs, batch_size=1, verbose=0)

    # Predicciones del modelo
    predictions = model.predict(x)
    
    return predictions