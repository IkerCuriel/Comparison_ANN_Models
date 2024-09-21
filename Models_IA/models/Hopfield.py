import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense

class Hopfield_Low_Level:
    def __init__(self, patterns):
        self.patterns = patterns
    
    def weights_matrix(self, patterns):
        p, n = patterns.shape
        weights = np.zeros((n, n))
        
        # Patterns - List of arrays
        '''
        patterns = np.array([
        array_1.flatten(),
        array_2.flatten(),
        array_3.flatten(),
        array_ruido_1.flatten(),
        array_ruido_2.flatten(),
        array_ruido_3.flatten(),
        ])

        patterns.shape
        '''

        for pt in patterns:
            pt = pt.reshape(-1, 1)
            weights += np.dot(pt, pt.T)

        np.fill_diagonal(weights, 0)
        weights /= n

        return weights

    def neuron(self, weights, inputs, epochs=1000):
        for i in range(epochs):
            inputs = np.sign(np.dot(weights, inputs))

        return inputs

    def reconstruct_pattern(self, noisy_pattern):
        weights = self.weights_matrix(self.patterns)
        reconstructed_pattern = self.neuron(weights, noisy_pattern.flatten())
        return reconstructed_pattern.reshape(noisy_pattern.shape)
    
def Hopfield_High_Level(input_dim):
    # Define el autoencoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(9, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Compila el modelo
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Entrena el autoencoder
    autoencoder.fit(Hopfield_Low_Level.self.patterns, Hopfield_Low_Level.self.patterns, epochs=100, batch_size=1, verbose=0)

    # Reconstruye los patrones
    reconstructed_patterns = autoencoder.predict(Hopfield_Low_Level.self.patterns)
    return reconstructed_patterns