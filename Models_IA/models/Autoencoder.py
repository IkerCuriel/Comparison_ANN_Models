import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

class Autoencoder_Low_Level:
    def __init__(self, x, alpha, momentum, epoch_max):
        self.x = x
        self.alpha = alpha
        self.momentum = momentum
        self.epoch_max = epoch_max
    
    def sigmoid(self, x):
        return float(1) / (float(1) + np.exp(-x))

    def sigmoid_dev(self, x):
        return self.sigmoid(x) * (float(1) - self.sigmoid(x))
    
    def autoencoder_img(self, x, alpha, momentum, epoch_max):
        inputs = x / 255.0
        inputs = inputs.flatten()
        expected_output = inputs

        input_neurons = inputs.shape[0] # 3 neurons
        hidden_neurons = 3
        output_neurons = input_neurons # 3 neurons

        weights_1 = 2 * np.random.random((input_neurons, hidden_neurons)) - 1
        weights_2 = 2 * np.random.random((hidden_neurons, output_neurons)) - 1

        w_old_1 = np.zeros_like(weights_1)
        w_new_1 = np.zeros_like(weights_1)

        w_new_2 = np.zeros_like(weights_2)
        w_old_2 = np.zeros_like(weights_2)

        epoch = 0
        loss = []
        mse = float(2)
        prev_mse = float(0)

        while (epoch < epoch_max) and abs(mse - prev_mse) > 0.00001:
            prev_mse = mse

            # FORWARD
            hidden_lyr_input = np.dot(inputs, weights_1)
            hidden_lyr_output = Autoencoder_Low_Level.self.sigmoid(hidden_lyr_input)

            # Capas de salida
            output_lyr_input = np.dot(hidden_lyr_output, weights_2)
            output = Autoencoder_Low_Level.self.sigmoid(output_lyr_input)

            # BACKPROPAGATION
            output_error = expected_output - output
            mse = np.mean((output_error)**2)
            loss.append(mse)
            gradient = output_error * Autoencoder_Low_Level.self.sigmoid_dev(output)

            output_lyr_delta = gradient

            # Calculo del error
            hidden_lyr_error = output_lyr_delta.dot(weights_2.T)
            hidden_lyr_delta = hidden_lyr_error * Autoencoder_Low_Level.sigmoid_dev(hidden_lyr_output)

            # Actualización de pesos
            w_new_2 = weights_2 + alpha * np.outer(hidden_lyr_output, output_lyr_delta) + momentum * (weights_2 - w_old_2)
            w_old_2 = weights_2.copy()
            weights_2 = w_new_2

            # Actualización de pesos
            w_new_1 = weights_1 + alpha * np.outer(inputs, hidden_lyr_delta) + momentum * (weights_1 - w_old_1)
            w_old_1 = weights_1.copy()
            weights_1 = w_new_1

            if (epoch % 200 == 0):
                print(f"Epoch: {epoch} Error: {mse}")

            epoch += 1

        latent_space = Autoencoder_Low_Level.self.sigmoid(np.dot(inputs, weights_1))
        decoded_inputs = Autoencoder_Low_Level.self.sigmoid(np.dot(latent_space, weights_2))
        decoded_inputs = (decoded_inputs * 255).astype(int)

        return loss, latent_space, decoded_inputs, weights_1, weights_2, Autoencoder_Low_Level.self.sigmoid


def Autoencoder_High_Level(image_array):
    # Obtener dimensiones de la imagen
    image_width, image_height, image_channels = image_array.shape

    # Paso 4: Define y entrena un autoencoder
    input_img = Input(shape=(image_width * image_height * image_channels,))
    
    # Codificador
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    
    # Decodificador
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(image_width * image_height * image_channels, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)

    # autoencoder.compile(optimizer=Adam(), loss='binary_crossentropy')

    # Entrenamiento
    autoencoder.fit(image_array.reshape(-1, image_width * image_height * image_channels),
                    image_array.reshape(-1, image_width * image_height * image_channels),
                    epochs=50, batch_size=32, shuffle=True)

    # Paso 5: Utiliza el autoencoder para reconstruir la imagen
    reconstructed_image_array = autoencoder.predict(image_array.reshape(-1, image_width * image_height * image_channels))

    # Reconstruir la imagen a su forma original
    reconstructed_image_array = reconstructed_image_array.reshape(image_width, image_height, image_channels)
    
    return reconstructed_image_array