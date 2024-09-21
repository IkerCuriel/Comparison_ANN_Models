import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt

class KohonenNetwork_low_level:
    def __init__(self, class_list, input, n_neurons, method, mode, alpha, epoch_max):
        self.input = input 
        self.class_list = class_list
        self.n_neurons = n_neurons
        self.method = method
        self.mode = mode
        self.alpha = alpha
        self.epoch_max = epoch_max

        # Definir pesos
        self.weights = np.random.rand(self.n_neurons, self.input)

    def normalize(self, data):
        w_min = np.min(data)
        w_max = np.max(data)
        norm_weights = (data - w_min) / (w_max - w_min)
        return norm_weights

    def train(self):
        norm_input =  self.normalize(self.class_list)
        norm_weights = self.normalize(self.weights)
        # pre_weights = np.copy(norm_weights)
        for _ in range(self.epoch_max):
            for input in norm_input:
                # Computing of the euclidean distance
                distances = np.linalg.norm(norm_weights - input, axis = 1)
                winner = np.argmin(distances)

                # Update of the winner neuron
                norm_weights[winner] += self.alpha * (input - norm_weights[winner])

                # Update the neighboring neurons
                for neighbor in range(self.n_neurons):
                    if self.method == 'direct':
                        # Calculate the distance from the winner neuron
                        distance_to_winner = np.linalg.norm(norm_weights[neighbor] - norm_weights[winner])

                        if distance_to_winner < 0.2:
                            if self.mode == 'alpha':
                                # Update the weights of neighbors using a Gaussian function
                                norm_weights[neighbor] += (self.alpha/2) * (input - norm_weights[neighbor])
                            elif self.mode == 'gaussian':
                                # Update the weights of neighbors using a Gaussian function
                                norm_weights[neighbor] += (self.alpha) * np.exp(-(distance_to_winner**2) / (2 * 1**2)) * (input - norm_weights[neighbor])
        
        return norm_weights
                                # 2         10      
                     
def KohonenNetwork_high_level(x, map_width, map_height): # Definir las dimensiones del mapa SOM
    som = MiniSom(map_width, map_height, 2, sigma=1.0, learning_rate=0.5)
    som.train_random(x, 100)  # Entrenamiento con 100 iteraciones
    weights = som.get_weights()
    
    def normalize(data):
        w_min = np.min(data)
        w_max = np.max(data)
        norm_weights = (data - w_min) / (w_max - w_min)
        return norm_weights  
    
    norm_weights = normalize(weights)
    norm_input = normalize(x)
    
    # Trazar los pesos de las neuronas en un scatterplot
    plt.scatter(norm_weights[:, :, 0], norm_weights[:, :, 1], c='b', marker='o', label='Neurons')
    # Trazar los datos normalizados en un scatterplot
    plt.scatter(norm_input[:, 0], norm_input[:, 1], c='r', marker='x', label='Data')
    plt.title('Kohonen Weights and Normalized Data')
    plt.legend()
    plt.show()
    
    return norm_weights, norm_input