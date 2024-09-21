import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

class LVQ_Low_Level:
    def norm_data(self, data):
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

    def init_vectors(self, data, y):
        num_classes = np.unique(y)
        selected_indices = [np.random.choice(np.where(y == label)[0]) for label in num_classes]
        normalized_vectors = np.array(data[selected_indices])
        return normalized_vectors

    def train(self, data, y, delta, epoch_max):
        vectors = self.init_vectors(data, y)
        for epoch in range(epoch_max):
            for data_point in range(len(data)):
                sample = data[data_point]
                output = y[data_point]

                distances = np.linalg.norm(vectors - sample, axis=1)
                sorted_distances_indices = np.argsort(distances)

                winner = sorted_distances_indices[0]
                second_winner = sorted_distances_indices[1]

                # Update winner and second winner
                if (y[winner] == output) and (y[second_winner] != output):
                    vectors[winner] += delta * (sample - vectors[winner])
                    vectors[second_winner] -= delta * (sample - vectors[second_winner])

                elif (y[winner] != output) and (y[second_winner] == output):
                    vectors[winner] -= delta * (sample - vectors[winner])
                    vectors[second_winner] += delta * (sample - vectors[second_winner])

        return vectors

    def test(self, data, vectors):
        winners = []
        for data_point in range(len(data)):
            sample = data[data_point]
            distances = np.linalg.norm(vectors - sample, axis=1)
            winner = np.argmin(distances)
            winners.append(winner)

        return winners
'''
    def plot(data, vectors, y, title, data_color='blue', vector_color='red'):
        with plt.style.context('seaborn-v0_8-darkgrid'):
            plt.scatter(data[:, 0], data[:, 1], c=y, cmap='viridis', marker='o', s=100, label='Data', edgecolor='black', linewidth=1)
            plt.scatter(vectors[:, 0], vectors[:, 1], c=vector_color, marker='x', s=200, label='Vectors', edgecolor='black', linewidth=1)
            plt.xlabel('characteristic x1')
            plt.ylabel('characteristic y1')
            plt.legend()
            plt.title(title)
            plt.show()

    def plot_2(data, vectors, y, title, data_color='blue', vector_color='red'):
        with plt.style.context('seaborn-v0_8-darkgrid'):
            plt.scatter(data[:, 0], data[:, 1], c=y, cmap='viridis', marker='o', s=100, label='Data', edgecolor='black', linewidth=1)
            plt.scatter(vectors[:, 0], vectors[:, 1], c=vector_color, marker='x', s=200, label='Vectors', edgecolor='black', linewidth=1)
            for i, t in enumerate(y):
                plt.annotate(t, (data[i, 0], data[i, 1]))
                plt.xlabel('characteristic x1')
                plt.ylabel('characteristic y1')
                plt.legend()
                plt.title(title)
                plt.show()
'''
                
def LVQ_High_Level(data, y, epochs=100, learning_rate=0.1):
    # Definición del modelo LVQ
    model_LVQ = Sequential()
    model_LVQ.add(Dense(4, input_dim=2, activation='sigmoid'))  # Capa oculta con 4 neuronas
    model_LVQ.add(Dense(1, activation='sigmoid'))  # Capa de salida con 1 neurona

    # Compilación del modelo LVQ
    sgd = SGD(learning_rate=learning_rate)
    model_LVQ.compile(loss='mean_squared_error', optimizer=sgd)

    # Entrenamiento del modelo LVQ
    model_LVQ.fit(data, y, epochs=epochs, batch_size=1, verbose=1)
    
    # Obtener los pesos de la capa oculta (vectores)
    vectors = model_LVQ.layers[0].get_weights()[0]

    # Evaluación del modelo LVQ
    scores = model_LVQ.evaluate(data, y)
    print("\n%s: %.2f%%" % (model_LVQ.metrics_names[0], scores*100))

    # Hacer predicciones con el modelo
    predictions = model_LVQ.predict(data)
    # Convertir las predicciones a valores binarios (0 o 1) basados en un umbral de 0.5
    binary_predictions = (predictions > 0.5).astype(int)

    # Calcular el accuracy comparando las predicciones con las etiquetas verdaderas
    accuracy = np.mean(binary_predictions.flatten() == y)
    print("Accuracy del modelo LVQ:", accuracy)

    return vectors, accuracy
