import tkinter as tk
import tkinter.ttk as ctk
from PIL import Image, ImageTk
import os
import random

class MainInterface(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Proyect - Iker Curiel')
        self.geometry('1000x500')
        self.resizable(False, False)
        
        # Obtener rutas de imágenes aleatorias
        adidas_img = self.get_random_image('Adidas')
        nike_img = self.get_random_image('Nike')
        converse_img = self.get_random_image('Converse')

        # Crear un marco para contener las imágenes
        self.frame_images = ctk.CTkFrame(master=self)
        self.frame_images.grid(row=0, pady=10, padx=0)

        # Mostrar las imágenes
        self.show_image(adidas_img, row=0, column=0)
        self.show_image(nike_img, row=0, column=1)
        self.show_image(converse_img, row=0, column=2)

    def get_random_image(self, brand):
        # Obtener una imagen aleatoria de la carpeta de la marca especificada
        images_path = f'/ruta/a/tu/carpeta/de/imagenes/{brand}/'  # Reemplaza con la ruta correcta
        images = os.listdir(images_path)
        random_image = random.choice(images)
        return os.path.join(images_path, random_image)

    def show_image(self, image_path, row, column):
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        image_label = ctk.CTkLabel(self.frame_images, image=photo)
        image_label.image = photo  # Mantener una referencia
        image_label.grid(row=row, column=column, padx=10, pady=10)

if __name__ == "__main__":
    app = MainInterface()
    app.mainloop()