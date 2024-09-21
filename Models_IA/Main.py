import tkinter as tk
from tkinter import messagebox, Label
from keras.layers import Input, Dense
from keras.models import Sequential, Model
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')
from tkinter import PhotoImage
from PIL import Image, ImageTk
import customtkinter as ctk
import numpy as np

# Import Models
from models import Autoencoder
from models import Backpropagation
from models import Hopfield
from models import LVQ_2
from models import Kohonen_Networks

class App(ctk.CTk):
    model_names = ["Kohonen Networks", "LVQ 2", "Autoencoder", "Hopfield", "Backpropagation"]
    model_levels = ["High Level", "Low Level"]
    
    def __init__(self):
        super().__init__()
        
        self.title('Models - Iker Curiel')
        self.geometry('1000x500')
        self.resizable(False, False)
        
        self.selected_model = ctk.StringVar(self)
        self.selected_level = ctk.StringVar(self)
        
        # Crear un marco para contener la imagen de fondo
        self.frame_0 = ctk.CTkFrame(master=self)
        self.frame_0.grid(row=0, pady=10, padx=0)

        # Agregar la imagen de fondo al marco
        background_image = PhotoImage(file="Assets/img/fondo.png")
        background_label = ctk.CTkLabel(self.frame_0, image=background_image)
        background_label.image = background_image  # Mantener una referencia
        background_label.grid(row=0, column=0)

        # Create widgets
        self.frame_1 = ctk.CTkFrame(master=self) # Creamos el marco.
        self.frame_1.grid(row=0, pady=10, padx=0) # Primera fila / relleno vertical / "" horizontal.

        # Create OptionMenu for model
        # self.model_optionmenu = ctk.CTkOptionMenu(self.frame_1, values=self.model_names, command=self.run_selected_model)
        self.model_optionmenu = ctk.CTkOptionMenu(self.frame_1, values=self.model_names, command=lambda value: self.run_selected_model(value))
        self.model_optionmenu.grid(row=0, column=0, pady=10, padx=10)
        
        # Create OptionMenu for level
        self.level_optionmenu = ctk.CTkOptionMenu(self.frame_1, values=self.model_levels, command=self.level_selected)
        self.level_optionmenu.grid(row=1, column=0, pady=10, padx=10)

        # Create a button to run the selected model
        # self.run_button = ctk.CTkButton(master=self.frame_1, text="Run Model", command=self.run_selected_model)
        self.run_button = ctk.CTkButton(master=self.frame_1, text="Run Model", command=lambda: self.run_selected_model(self.selected_model.get()))
        self.run_button.grid(row=2, column=0, columnspan=len(self.model_names), pady=10, padx=10)
        
        # Create a button to exit the program
        self.close_app = ctk.CTkButton(master=self.frame_1, text="Close App", command=self.exit_app)
        self.close_app.grid(row=3, column=0, columnspan=len(self.model_names)+1, sticky='we')
        
        # Crear un marco para contener el gráfico
        # self.graph_frame = self.frame_0(self)
        # self.graph_frame.grid(row=3, column=0, columnspan=len(self.model_names), pady=10, padx=10)
        
    def show_image_result(self, image_path):
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        self.result_label.configure(image=photo)
        self.result_label.image = photo  # Esto evita que la imagen sea eliminada por el recolector de basura
        
    def print_result(self, result):
        self.result_text = tk.Text(master=self.frame_1, width=50, height=10)
        self.result_text.grid(row=4, column=0, columnspan=len(self.model_names)+1, sticky='we')
        self.result_text.insert(tk.END, result)
        
    def model_selected(self, value, model_index):
        self.selected_model.set(value)

    def level_selected(self, value):
        model_level = value
        self.selected_level.set(model_level)
        print("Selected model level:", model_level)        
        '''
        self.combobox_1 = ctk.CTkComboBox(frame_1, values=["Option 1", "Option 2", "Option 3"])
        self.combobox_1.grid(row=0, column=1, pady=10, padx=0) 
        self.combobox_1.set("LVQ 2") 

        appearance_mode_optionemenu = ctk.CTkOptionMenu(frame_1, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event)
        appearance_mode_optionemenu.grid(row=0, column=2, padx=0, pady=10)

        scaling_optionemenu = ctk.CTkOptionMenu(frame_1, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling_event)
        scaling_optionemenu.grid(row=0, column=3, padx=0, pady=10)
        
        text_1 = ctk.CTkTextbox(master=frame_1, width=1125, height=70)
        text_1.grid(row=1, column=0, columnspan=3, pady=3, padx=10)
        text_1.insert("0.0", "Lorem Ipsum...")
        '''  

    #===============    MODELS    ====================
    #----------  AUTOENCODER LOW LEVEL  --------------
    def run_autoencoder_low_level(self, x=None):
        print("Ejecutando run_autoencoder_low_level")
        
        if x is not None: 
            imagen_original = x
            # Nuevas entradas
            '''
            # Abrir la imagen original
            imagen_original = Image.open(x)
            # Redimensiona la imagen a 7x7 píxeles utilizando LANCZOS
            imagen_redimensionada = imagen_original.resize((7, 7), Image.LANCZOS)
            imagen_redimensionada.save("imagen_redimensionada.jpg")
            messagebox.showinfo("Resultado", "Autoencoder ejecutado correctamente.")
            # Carga la imagen de entrada
            input_image = Image.open("/content/imagen_redimensionada.jpg")  # 7x7 píxeles
            width, height = input_image.size
            # Convert the image to a NumPy array and reshape it
            image_array = np.array(input_image)
            image_array = image_array.reshape(-1, 3)
            '''
            
        else:
            # Abrir la imagen original
            print("Cargando imagen original")  # Agregar para verificar si se está cargando la imagen
            imagen_original = Image.open("Assets/Autoencoder_img/Carita.jpeg")
            # Redimensiona la imagen a 7x7 píxeles utilizando LANCZOS
            imagen_redimensionada = imagen_original.resize((7, 7), Image.LANCZOS)
            imagen_redimensionada.save("imagen_redimensionada.jpg")
            messagebox.showinfo("Resultado", "Autoencoder ejecutado correctamente.")  # Verificar si se muestra el mensaje de información
            # Carga la imagen de entrada
            input_image = Image.open("/content/imagen_redimensionada.jpg")  # 7x7 píxeles
            width, height = input_image.size
            # Convert the image to a NumPy array and reshape it
            image_array = np.array(input_image)
            image_array = image_array.reshape(-1, 3)

            data = np.array(image_array)
            alpha = 1
            momentum = 0.2
            epoch_max = 1000

            # Definición del modelo en Keras
            model = Autoencoder.Autoencoder_Low_Level(data, alpha, momentum, epoch_max)
            loss, latent_space, decoded_inputs, trained_weights_1, trained_weights_2, trained_sigmoid = model

            # Visualización de los resultados
            # Capturar los resultados en variables de cadena
            data_flatten_str = np.array2string(data.flatten())
            latent_space_str = np.array2string(latent_space)
            decoded_inputs_str = np.array2string(decoded_inputs)

            # Mostrar la reconstrucción de la imagen
            reconstructed_image = Image.fromarray(np.uint8(decoded_inputs.reshape(height, width, 3)))
            reconstructed_photo = ImageTk.PhotoImage(reconstructed_image)

            canvas = ctk.CTkCanvas(self.frame_1, width=reconstructed_image.width, height=reconstructed_image.height)
            canvas.grid(row=2, column=0, columnspan=3)
            
            # Asegúrate de guardar una referencia al objeto PhotoImage
            canvas.reconstructed_photo = reconstructed_photo

            canvas.create_image(0, 0, anchor="nw", image=reconstructed_photo)

            # Crear etiquetas para mostrar otros resultados
            label_data = ctk.CTkLabel(self.frame_1, text=f"Data flatten:\n{data_flatten_str}")
            label_data.grid(row=1, column=0, pady=5)

            label_latent_space = ctk.CTkLabel(self.frame_1, text=f"Latent space:\n{latent_space_str}")
            label_latent_space.grid(row=1, column=1, pady=5)

            label_decoded_inputs = ctk.CTkLabel(self.frame_1, text=f"Decoded inputs:\n{decoded_inputs_str}")
            label_decoded_inputs.grid(row=1, column=2, pady=5)
           
    #----------  AUTOENCODER HIGH LEVEL  --------------
    def run_autoencoder_high_level(self, x=None):
        if x is not None: 
            imagen_original = x
            # Nuevas entradas
            '''
            # Abrir la imagen original
            imagen_original = Image.open(x)
            # Redimensiona la imagen a 7x7 píxeles utilizando LANCZOS
            imagen_redimensionada = imagen_original.resize((7, 7), Image.LANCZOS)
            imagen_redimensionada.save("imagen_redimensionada.jpg")
            messagebox.showinfo("Resultado", "Autoencoder ejecutado correctamente.")
            # Carga la imagen de entrada
            input_image = Image.open("/content/imagen_redimensionada.jpg")  # 7x7 píxeles
            width, height = input_image.size
            # Convert the image to a NumPy array and reshape it
            image_array = np.array(input_image)
            image_array = image_array.reshape(-1, 3)
            '''
            
        else:
            # Abrir la imagen original
            imagen_original = Image.open("Assets/Autoencoder_img/Carita.jpeg")
            # Convertimos la imagen a un arreglo numpy
            image_array = np.array(imagen_original)
            # Normalizamos los valores de los píxeles a un rango entre 0 y 1
            image_array = image_array.astype('float32') / 255.0
            
            # Definición del modelo en Keras
            model = Autoencoder.Autoencoder_High_Level(image_array)
            reconstructed_image_array = model
            
            # Paso 6: Mostrar la imagen original y la imagen reconstruida
            plt.figure(figsize=(10, 5))

            # Imagen original
            plt.subplot(1, 2, 1)
            plt.imshow(imagen_original)
            plt.title('Imagen Original')
            plt.axis('off')

            # Imagen reconstruida
            reconstructed_image = Image.fromarray((reconstructed_image_array * 255).astype(np.uint8))
            plt.subplot(1, 2, 2)
            plt.imshow(reconstructed_image)
            plt.title('Imagen Reconstruida por el Autoencoder')
            plt.axis('off')
            
            plt.show()

    #----------  BACKPROPAGATION LOW LEVEL  --------------
    def run_backpropagation_low_level(self, x=None):
        if x is not None: 
            imagen_original = x
            # Nuevas entradas
                    
        else:
            # Parámetros self, x, y, l_r, alpha, epochs
            x = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
            y = np.array([[0], [1], [1], [0]])
            learning_rate = 0.1
            alpha = 0.5  # momentum
            epochs = 100
            
            # Definición del modelo en Keras
            model = Backpropagation.Backpropagation_Low_Level(x, y, learning_rate, alpha, epochs)
            grafico1 = model.generar_grafico1()
            # Convertir los gráficos a formatos que CustomTkinter pueda mostrar
            imagen_grafico1 = ImageTk.PhotoImage(grafico1)
            # Crear etiquetas en la ventana para mostrar los gráficos
            etiqueta_grafico1 = ctk.CTkLabel(self, image=imagen_grafico1)
            etiqueta_grafico1.image = imagen_grafico1
            etiqueta_grafico1.pack()      
    
    #----------  BACKPROPAGATION HIGH LEVEL  --------------
    def run_backpropagation_high_level(self, x=None):
        if x is not None: 
            imagen_original = x
            # Nuevas entradas
                    
        else:
            # Parámetros self, x, y, l_r, alpha, epochs
            x = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
            y = np.array([[0], [1], [1], [0]])
            
            # Definición del modelo en Keras
            model = Sequential()
            model.add(Dense(1, input_dim=3, activation='sigmoid'))

            # Compilación del modelo
            model.compile(loss='mean_squared_error', optimizer='sgd')

            # Entrenamiento del modelo
            model.fit(x, y, epochs=100, batch_size=1, verbose=0)

            # Predicciones del modelo
            predictions = model.predict(x)

            # Convertir arrays numpy a strings para mostrar en etiquetas
            salida_deseada_str = np.array2string(y)
            salida_obtenida_str = np.array2string(predictions)

            # Crear etiquetas para mostrar los resultados
            label_deseada = ctk.CTkLabel(self, text=f"Salida deseada (y):\n{salida_deseada_str}")
            label_deseada.grid(row=1, column=0)

            label_obtenida = ctk.CTkLabel(self, text=f"Salida obtenida (yo):\n{salida_obtenida_str}")
            label_obtenida.grid(row=1, column=1)
            
    #----------  HOPFIELD LOW LEVEL  --------------
    def run_hopfield_low_level(self, x=None):
        if x is not None:
            x = x
        else:
            array_1 = np.array([
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1]])
            array_ruido_1 = np.array([
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, 1, -1, 1, 1, 1],
                [1, -1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, -1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, -1],
                [1, 1, 1, -1, 1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, -1],
                [1, 1, 1, -1, -1, -1, 1, -1, 1]])
            array_2 = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1]])
            array_ruido_2 = np.array([
                [-1, 1, 1, 1, 1, 1, 1, 1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, 1, 1, 1, 1, 1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1, -1, -1, -1]])
            array_3 = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, -1, -1, -1]])
            array_ruido_3 = np.array([
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, 1, 1, 1, 1, 1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, -1, -1, -1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1]])
            array_4 = np.array([
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1]])
            array_ruido_4 = np.array([
                [1,1,1,1,-1,-1,1,1,1],
                [1,1,1,1,-1,-1,1,1,1],
                [1,1,1,1,1,-1,1,-1,1],
                [1,-1,-1,-1,1,1,-1,-1,1],
                [1,1,1,1,-1,-1,-1,-1,1],
                [-1,1,1,1,-1,-1,-1,-1,1],
                [-1,1,-1,1,1,-1,1,1,1],
                [-1,-1,1,1,1,1,1,1,1],
                [1,1,1,1,-1,1,1,-1,1]])
            array_5 = np.array([
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1]])
            array_ruido_5 = np.array([
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, 1, 1, 1]])
            
            patterns = np.array([
                array_1.flatten(),
                array_2.flatten(),
                array_3.flatten(),
                array_ruido_1.flatten(),
                array_ruido_2.flatten(),
                array_ruido_3.flatten(),
                ])

            model = Hopfield.Hopfield_Low_Level(patterns)
            weights, inputs = model
            
            # Genera los patrones reconstruidos
            fig = plt.figure(figsize=(7, 7))
            for i, array_ruido in enumerate([array_ruido_1], start=1):  # Aquí puedes poner todos tus arrays de ruido
                rc_pattern = model.neuron(weights, array_ruido.flatten())
                rc_pattern = rc_pattern.reshape(9, 9)
                
                ax = fig.add_subplot(1, 1, i)
                ax.imshow(rc_pattern, cmap='gray')
                ax.set_title(f"Pattern reconstructed {i}")

            fig.tight_layout()
            
    #----------  HOPFIELD HIGH LEVEL  --------------
    def run_hopfield_high_level(self, x=None):
        if x is not None:
            self.x = x
        else:
            array_1 = np.array([
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1]])
            array_ruido_1 = np.array([
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, 1, -1, 1, 1, 1],
                [1, -1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, -1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, -1],
                [1, 1, 1, -1, 1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, -1],
                [1, 1, 1, -1, -1, -1, 1, -1, 1]])
            array_2 = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1]])
            array_ruido_2 = np.array([
                [-1, 1, 1, 1, 1, 1, 1, 1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, 1, 1, 1, 1, 1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, -1, 1, -1, -1, -1, -1]])
            array_3 = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, -1, -1, -1]])
            array_ruido_3 = np.array([
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-1, -1, 1, 1, 1, 1, 1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, 1, -1, -1, -1],
                [1, 1, 1, 1, 1, -1, -1, -1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1]])
            array_4 = np.array([
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1],
                [-1, -1, -1, 1, 1, 1, -1, -1, -1]])
            array_ruido_4 = np.array([
                [1,1,1,1,-1,-1,1,1,1],
                [1,1,1,1,-1,-1,1,1,1],
                [1,1,1,1,1,-1,1,-1,1],
                [1,-1,-1,-1,1,1,-1,-1,1],
                [1,1,1,1,-1,-1,-1,-1,1],
                [-1,1,1,1,-1,-1,-1,-1,1],
                [-1,1,-1,1,1,-1,1,1,1],
                [-1,-1,1,1,1,1,1,1,1],
                [1,1,1,1,-1,1,1,-1,1]])
            array_5 = np.array([
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1]])
            array_ruido_5 = np.array([
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [1, 1, 1, -1, -1, -1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, 1, 1, 1],
                [-1, -1, -1, -1, -1, -1, 1, 1, 1]])
            
            patterns = np.array([
                array_1.flatten(),
                array_2.flatten(),
                array_3.flatten(),
                array_ruido_1.flatten(),
                array_ruido_2.flatten(),
                array_ruido_3.flatten(),
                ])
            
            # Define la dimensión de entrada para el autoencoder
            input_dim = patterns.shape[1]

            # Define el autoencoder
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(9, activation='relu')(input_layer)
            decoded = Dense(input_dim, activation='sigmoid')(encoded)

            # Compila el modelo
            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

            # Entrena el autoencoder
            autoencoder.fit(patterns, patterns, epochs=100, batch_size=1, verbose=0)

            # Reconstruye los patrones
            reconstructed_patterns = autoencoder.predict(patterns)
            
            # Muestra los resultados
            fig = plt.figure(figsize=(10, 5))
            for i in range(len(patterns)):
                ax = fig.add_subplot(2, len(patterns)//2, i+1)
                ax.imshow(reconstructed_patterns[i].reshape((9, 9)), cmap='gray')
                ax.set_title(f'Patrón {i+1}')
                ax.axis('off')
            plt.tight_layout()
            plt.show()
 
    #----------  LVQ2 LOW LEVEL  --------------
    def run_lvq2_low_level(self, x=None):
        if x is not None:
            self.x = x
        else:
            data = np.array([[5.2, 6.7], [2.0, 3.0], [4.0, 5.0], [7.9, 6.1], [1.0, 2.0], [7.1, 8.9],
                 [2.0, 1.0], [5.1, 7.2], [3.0, 3.0], [7.9, 5.2], [4.0, 5.0], [2.9, 3.2],

                 [4.1, 5.7], [1.2, 2.8], [7.9, 6.2], [5.2, 6.1], [9.2, 8.1], [2.2, 1.1],
                 [3.9, 2.0], [4.8, 5.2], [9.9, 8.2], [7.1, 5.7], [5.2, 7.9], [7.2, 8.1]])

            data_new_2 = np.array([[1.1, 2.5], [3.1, 2.3], [1.1, 2.1], [3.1, 2.9], [3.2, 3.7], [3.4, 1.6],
                                [2.2, 1.1], [3.6, 1.5], [3.4, 1.1], [4.3, 1.3], [2.9, 4.5], [1.1, 2.9],

                                [5.7, 9.1], [9.7, 7.2], [6.3, 7.4], [6.6, 7.4], [5.7, 8.7], [5.4, 8.6],
                                [9.3, 8.9], [7.1, 9.5], [6.4, 8.8], [8.6, 8.7], [8.2, 9.1], [8.3, 9.3]])

            x_data = np.array([[5.9, 6.2], [2.6, 3.2], [4.8, 5.1], [1.7, 2.2], [2.9,1.5], [3.8, 3.1],
                            [4.5, 5.2], [9.0, 8.0], [7.2, 5.9], [5.1, 7.8], [7.2, 8.9], [7.1, 6.3]])

            y_data = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

            y = np.array([1, 0, 0, 1, 0, 1,
                        0, 1, 0, 1, 0, 0,
                        0, 0, 1, 1, 1, 0,
                        0, 0, 1, 1, 1, 1])

            y_op_2 = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
            
            # Define la función de trazado
            def plot(data, vectors, y, title, data_color='blue', vector_color='red'):
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
            
            model = LVQ_2.LVQ_Low_Level()
            norm_data = model.norm_data(data_new_2)
            norm_vectors = model.init_vectors(norm_data, y)
            plot(norm_data, norm_vectors, y, title='Before the training')
        
    #----------  LVQ2 HIGH LEVEL  --------------
    def run_lvq2_high_level(self, x=None):
        if x is not None:
            self.x = x
        else:
            data = np.array([[5.2, 6.7], [2.0, 3.0], [4.0, 5.0], [7.9, 6.1], [1.0, 2.0], [7.1, 8.9],
                 [2.0, 1.0], [5.1, 7.2], [3.0, 3.0], [7.9, 5.2], [4.0, 5.0], [2.9, 3.2],

                 [4.1, 5.7], [1.2, 2.8], [7.9, 6.2], [5.2, 6.1], [9.2, 8.1], [2.2, 1.1],
                 [3.9, 2.0], [4.8, 5.2], [9.9, 8.2], [7.1, 5.7], [5.2, 7.9], [7.2, 8.1]])

            data_new_2 = np.array([[1.1, 2.5], [3.1, 2.3], [1.1, 2.1], [3.1, 2.9], [3.2, 3.7], [3.4, 1.6],
                                [2.2, 1.1], [3.6, 1.5], [3.4, 1.1], [4.3, 1.3], [2.9, 4.5], [1.1, 2.9],

                                [5.7, 9.1], [9.7, 7.2], [6.3, 7.4], [6.6, 7.4], [5.7, 8.7], [5.4, 8.6],
                                [9.3, 8.9], [7.1, 9.5], [6.4, 8.8], [8.6, 8.7], [8.2, 9.1], [8.3, 9.3]])

            x_data = np.array([[5.9, 6.2], [2.6, 3.2], [4.8, 5.1], [1.7, 2.2], [2.9,1.5], [3.8, 3.1],
                            [4.5, 5.2], [9.0, 8.0], [7.2, 5.9], [5.1, 7.8], [7.2, 8.9], [7.1, 6.3]])

            y_data = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

            y = np.array([1, 0, 0, 1, 0, 1,
                        0, 1, 0, 1, 0, 0,
                        0, 0, 1, 1, 1, 0,
                        0, 0, 1, 1, 1, 1])

            y_op_2 = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
            
            # Define la función de trazado
            def plot(data, vectors, y, title, data_color='blue', vector_color='red'):
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
            
            model = LVQ_2.LVQ_High_Level(data_new_2, y, epochs=100, learning_rate=0.1)
            vectors, accuracy = model
            plot(data_new_2, vectors, y, 'LVQ Model')
                      
    #----------  KOHONEN NETWORKS LOW LEVEL  --------------
    def run_kohonen_networks_low_level(self, x=None):
        if x is not None:
            self.x = x
        else:
            print("Dentro de run_kohonen_networks_high_level")
            # Define 20 points in 4 different classes
            class_1 = np.array([[1, 2], [0.5, 2], [0.5, 0.5], [1, 0.9], [0.5, 1.2],
                    [0.8, 1.4], [1, 1.6], [1.4, 0.6], [1, 0.9], [1, 0.9],
                    [1.8, 1.4], [2, 2], [1, 2], [1, 0.5], [1.5, 0.9],
                    [0.5, 1.3], [1, 0.8], [1.2, 0.5], [0.5, 1.8], [0.9, 0.7]])

            class_2 = np.array([[5, 5], [4.5, 4.7], [5, 4.4], [4.1, 3.9], [4.6, 4.2],
                                [4, 4.5], [4.3, 5], [4.7, 4.1], [4.4, 3.8], [4.3, 4.9],
                                [4.2, 4.7], [3.4, 4.1], [4.2, 3.9], [4.5, 4.6], [4, 3.5],
                                [5, 4.2], [3.1, 4.3], [4, 4.4], [4.2, 3.8], [4.6, 3.9]])

            class_3 = np.array([[1, 5], [1.2, 4.8], [1.1, 4.5], [3.9, 4.51], [1.7, 4.9],
                                [2, 3.8], [1.6, 4.2], [1.2, 3.9], [0.7, 3.7], [1.6, 3.6],
                                [1.1, 4.2], [1.2, 4.8], [1.3, 3.5], [0.2, 4.2], [1.8, 4.2],
                                [1.6, 3.2], [2, 4.2], [1.4, 4.5], [1.6, 4.2], [0.4, 3.9]])

            class_4 = np.array([[4.5, 0.9], [4.9, 1.2], [4.7, 0.5], [3.9, 1.1], [4.2, 0.9],
                                [3.4, 1.2], [3.7, 1.9], [4.1, 1.5], [3.6, 1.6], [3.8, 2.1],
                                [3.2, 1.7], [3.2, 0.7], [3, 2.2], [3.5, 2.2], [3.6, 0.5],
                                [3.5, 2.3], [3.9, 0.6], [3.1, 1.3], [3.7, 1.3], [3.3, 1.9]])

            # ndarray with shape (80, 2)
            class_list = np.vstack((class_1, class_2, class_3, class_4))

            # Define the number of neurons
            method = 'direct'
            mode = 'alpha'
            n_neurons = 20
            alpha = 0.7
            epoch_max = 30

            # Create an instance of KohonenNetwork_low_level
            model = Kohonen_Networks.KohonenNetwork_low_level(class_list, 2, n_neurons, method, mode, alpha, epoch_max)

            # Train the model
            norm_weights = model.train()

            # Plot the neurons
            plt.figure(figsize=(8, 6))
            plt.scatter(norm_weights[:, 0], norm_weights[:, 1], c='b', marker='o', label='Neurons')
            plt.title('Kohonen Weights')
            plt.legend()
            plt.show()
            
    #----------  KOHONEN NETWORKS HIGH LEVEL  --------------
    def run_kohonen_networks_high_level(self, x=None):
        if x is not None:
            self.x = x
        else:
            # Define 20 points in 4 different classes
            class_1 = np.array([[1, 2], [0.5, 2], [0.5, 0.5], [1, 0.9], [0.5, 1.2],
                                [0.8, 1.4], [1, 1.6], [1.4, 0.6], [1, 0.9], [1, 0.9],
                                [1.8, 1.4], [2, 2], [1, 2], [1, 0.5], [1.5, 0.9],
                                [0.5, 1.3], [1, 0.8], [1.2, 0.5], [0.5, 1.8], [0.9, 0.7]])

            class_2 = np.array([[5, 5], [4.5, 4.7], [5, 4.4], [4.1, 3.9], [4.6, 4.2],
                                [4, 4.5], [4.3, 5], [4.7, 4.1], [4.4, 3.8], [4.3, 4.9],
                                [4.2, 4.7], [3.4, 4.1], [4.2, 3.9], [4.5, 4.6], [4, 3.5],
                                [5, 4.2], [3.1, 4.3], [4, 4.4], [4.2, 3.8], [4.6, 3.9]])

            class_3 = np.array([[1, 5], [1.2, 4.8], [1.1, 4.5], [3.9, 4.51], [1.7, 4.9],
                                [2, 3.8], [1.6, 4.2], [1.2, 3.9], [0.7, 3.7], [1.6, 3.6],
                                [1.1, 4.2], [1.2, 4.8], [1.3, 3.5], [0.2, 4.2], [1.8, 4.2],
                                [1.6, 3.2], [2, 4.2], [1.4, 4.5], [1.6, 4.2], [0.4, 3.9]])

            class_4 = np.array([[4.5, 0.9], [4.9, 1.2], [4.7, 0.5], [3.9, 1.1], [4.2, 0.9],
                                [3.4, 1.2], [3.7, 1.9], [4.1, 1.5], [3.6, 1.6], [3.8, 2.1],
                                [3.2, 1.7], [3.2, 0.7], [3, 2.2], [3.5, 2.2], [3.6, 0.5],
                                [3.5, 2.3], [3.9, 0.6], [3.1, 1.3], [3.7, 1.3], [3.3, 1.9]])

            # ndarray with shape (80, 2)
            class_list = np.vstack((class_1, class_2, class_3, class_4))
            
            def plot_som_weights_and_data(norm_weights, norm_input):
                plt.figure(figsize=(8, 6))
                plt.scatter(norm_weights[:, :, 0], norm_weights[:, :, 1], c='b', marker='o', label='Neurons')
                plt.scatter(norm_input[:, 0], norm_input[:, 1], c='r', marker='x', label='Data')
                plt.title('Pesos de Kohonen y Datos Normalizados')
                plt.legend()
                plt.xlabel('Característica x1')
                plt.ylabel('Característica y1')
                plt.show()
            
            model = Kohonen_Networks.KohonenNetwork_high_level(class_list, 2, 10)
            norm_weights, norm_input = model
            plot_som_weights_and_data(norm_weights, norm_input)           
              
    #===============    BUTTONS COMMANDS    ====================
    
    '''
    def run_selected_model(self):
        model_name = self.selected_model.get()
        model_level = self.selected_level.get()
        
        # Imprimir los valores para depurar
        print("Model Name:", model_name)
        print("Model Level:", model_level)
        
        # model_names = ["Kohonen Networks", "LVQ 2", "Autoencoder", "Hopfield", "Backpropagation"]
        if model_name == "Autoencoder":
            if model_level == "High Level":
                print("Running Autoencoder in High Level")
                self.run_autoencoder_high_level()
            elif model_level == "Low Level":
                print("Running Autoencoder in Low Level")
                self.run_autoencoder_low_level()
            else:
                print("Invalid model level selected:", model_level)
        else:
            print("Invalid model selected:", model_name)
            
        # Ejemplo de cómo mostrar resultados:
        result = f"Ejecutando {model_name} en {model_level} Level"
        self.print_result(result)
    '''      
    
    def run_selected_model(self, value, *args):
        model_name = value
        model_level = self.selected_level.get()
        
        if model_level not in self.model_levels:
            print("Invalid model level selected:", model_level)
            return
        
        print("Model Name:", model_name)
        print("Model Level:", model_level)

        if model_name == "Autoencoder":
            if model_level == "High Level":
                print("Running Autoencoder in High Level")
                self.run_autoencoder_high_level()
            elif model_level == "Low Level":
                print("Running Autoencoder in Low Level")
                self.run_autoencoder_low_level()
        else:
            print("Invalid model selected:", model_name)
            
        # Example of how to show results
        result = f"Running {model_name} in {model_level} Level"
        self.print_result(result)
        
        if model_name == "Kohonen Networks":
            if model_level == "High Level":
                self.run_kohonen_networks_high_level()
                # Código para correr Kohonen Networks en High Level
            elif model_level == "Low Level":
                self.run_kohonen_networks_low_level()
                # Código para correr Kohonen Networks en Low Level
                # Ejemplo de cómo mostrar resultados en una ventana emergente:
                if model_name == "Kohonen Networks" and model_level == "High Level":
                    model = Kohonen_Networks.KohonenNetwork_high_level(self.class_list, 2, 10)
                    norm_weights, norm_input = model
                    self.show_result_window()
                    self.plot_som_weights_and_data(norm_weights, norm_input)
               
        if model_name == "LVQ 2":
            if model_level == "High Level":
                self.run_lvq2_high_level()
                # Código para correr Kohonen Networks en High Level
                print("Corriendo LVQ 2 en High Level")
            elif model_level == "Low Level":
                self.run_lvq2_low_level()
                # Código para correr Kohonen Networks en Low Level
                print("Corriendo LVQ 2 en Low Level")
            # Ejemplo de cómo mostrar resultados:
            result = f"Ejecutando {model_name} en {model_level} Level"
            self.print_result(result)
        
        if model_name == "Hopfield":
            if model_level == "High Level":
                self.run_hopfield_high_level()
                # Código para correr Kohonen Networks en High Level
                print("Corriendo Hopfield en High Level")
            elif model_level == "Low Level":
                self.run_hopfield_low_level()
                # Código para correr Kohonen Networks en Low Level
                print("Corriendo Hopfield en Low Level")
            # Ejemplo de cómo mostrar resultados:
            result = f"Ejecutando {model_name} en {model_level} Level"
            self.print_result(result)
                
        if model_name == "Backpropagation":
            if model_level == "High Level":
                self.run_backpropagation_high_level()
                # Código para correr Kohonen Networks en High Level
                print("Corriendo Backpropagation en High Level")
            elif model_level == "Low Level":
                self.run_backpropagation_low_level()
                # Código para correr Kohonen Networks en Low Level
                print("Corriendo Backpropagation en Low Level")
            # Ejemplo de cómo mostrar resultados:
            result = f"Ejecutando {model_name} en {model_level} Level"
            self.print_result(result)
            
    def exit_app(self):
        self.destroy()
    
    def button_callback(self):
        print("Button click", self.combobox_1.get())
    
    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)
    
    def change_scaling_event(self, new_scaling: str): 
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)
  
if __name__ ==  "__main__":
    app = App()
    app.mainloop()