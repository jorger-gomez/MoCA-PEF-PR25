import os
from tkinter import Tk, filedialog
from PIL import Image, ImageOps, ImageChops

def process_images():
    # Abrir ventana para seleccionar la carpeta
    Tk().withdraw()  # Oculta la ventana principal de Tkinter
    folder_path = filedialog.askdirectory(title="Selecciona una carpeta")
    
    if not folder_path:
        print("No se seleccionó ninguna carpeta.")
        return

    # Crear carpetas de salida
    padded_folder = os.path.join(folder_path, "cuadradas")  # Para imágenes con padding
    resized_folder = os.path.join(folder_path, "redimensionadas")  # Para imágenes redimensionadas
    os.makedirs(padded_folder, exist_ok=True)
    os.makedirs(resized_folder, exist_ok=True)

    # Iterar sobre las imágenes en la carpeta
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    # Convertir a modo RGB si es necesario
                    img = img.convert("RGB")

                    # Detectar y recortar bordes blancos
                    bg = Image.new(img.mode, img.size, (255, 255, 255))  # Fondo blanco
                    diff = ImageChops.difference(img, bg)
                    bbox = diff.getbbox()
                    if bbox:
                        img = img.crop(bbox)

                    # Obtener el tamaño máximo para el padding
                    max_side = max(img.size)
                    new_img = Image.new("RGB", (max_side, max_side), (255, 255, 255))  # Fondo blanco

                    # Centrar la imagen original en el fondo
                    offset = ((max_side - img.size[0]) // 2, (max_side - img.size[1]) // 2)
                    new_img.paste(img, offset)

                    # Guardar la imagen con padding
                    padded_output_path = os.path.join(padded_folder, filename)
                    new_img.save(padded_output_path)
                    print(f"Imagen con padding guardada: {padded_output_path}")

                    # Redimensionar a 128x128 píxeles
                    resized_img = new_img.resize((128, 128), Image.Resampling.LANCZOS)

                    # Guardar la imagen redimensionada
                    resized_output_path = os.path.join(resized_folder, filename)
                    resized_img.save(resized_output_path)
                    print(f"Imagen redimensionada guardada: {resized_output_path}")
            except Exception as e:
                print(f"No se pudo procesar el archivo {filename}: {e}")

if __name__ == "__main__":
    process_images()
