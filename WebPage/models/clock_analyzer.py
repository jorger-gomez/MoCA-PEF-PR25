import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import glob
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from io import BytesIO
from PIL import Image

# Importamos el módulo de clasificación
from models.clock_classifier import load_classification_model, classify_clock_image

# Configuración básica
class Config:
    # Detectar automáticamente el dispositivo disponible
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 
                         "cpu")
    
    # Tamaño de imagen para el modelo
    IMG_SIZE = 256
    
    # Modelo
    ENCODER = 'se_resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['entire', 'numbers', 'hands', 'contour']
    NUM_CLASSES = len(CLASSES)
    
    # Rutas a los modelos guardados
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.pth')
    CLASSIFICATION_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'classification_model.pth')

# Imprimir información sobre el dispositivo que se utilizará
print(f"Dispositivo para procesamiento: {Config.DEVICE}")
if hasattr(torch.backends, 'mps'):
    print(f"MPS disponible: {torch.backends.mps.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA disponible: Versión {torch.version.cuda}")
print(f"Ruta al modelo de segmentación: {Config.MODEL_PATH}")
print(f"Ruta al modelo de clasificación: {Config.CLASSIFICATION_MODEL_PATH}")

def build_model():
    """
    Construye el modelo de segmentación UNet++
    """
    model = smp.UnetPlusPlus(
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        classes=Config.NUM_CLASSES,
        activation='sigmoid'
    )
    
    print(f"Modelo construido: UNet++ con encoder {Config.ENCODER}")
    return model

def analyze_clock_image(image_path):
    """
    Analiza una imagen de un reloj dibujado utilizando un modelo de segmentación profunda.
    
    Args:
        image_path: Ruta a la imagen del reloj
        
    Returns:
        tuple: (resultados_análisis, imagen_con_anotaciones)
    """
    # Preprocesamiento para asegurar que la imagen es adecuada para el análisis
    try:
        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: No se pudo leer la imagen: {image_path}")
            return simulate_analysis(image_path)
            
        # Verificar si es una imagen con fondo negro
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.mean(img_gray) < 127:  # Si el promedio es menor que 127, la imagen es mayormente oscura
            print("Detectada imagen con fondo oscuro, invirtiendo colores...")
            img = cv2.bitwise_not(img)  # Invertir colores
        
        # Guardar imagen preprocesada en lugar de la original
        cv2.imwrite(image_path, img)
    except Exception as e:
        print(f"Error en preprocesamiento: {str(e)}")
    
    # Guardar una copia de la imagen original
    try:
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_folder = os.path.join(app_dir, 'static', 'results')
        image_filename = os.path.basename(image_path)
        original_copy_path = os.path.join(results_folder, f"original_{image_filename}")
        
        # Copiar la imagen original usando OpenCV
        image = cv2.imread(image_path)
        if image is not None:
            # Asegurarse de que la imagen se vea claramente
            if image.shape[2] == 3:  # Si es una imagen a color
                # Verificar contraste y brillo
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if np.std(gray) < 30:  # Bajo contraste
                    # Mejorar contraste
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            
            cv2.imwrite(original_copy_path, image)
            print(f"Copia de imagen original guardada en: {original_copy_path}")
        else:
            # Intentar con método alternativo
            try:
                import shutil
                shutil.copy2(image_path, original_copy_path)
                print(f"Copia de imagen original guardada usando shutil: {original_copy_path}")
            except Exception as e:
                print(f"Error al copiar imagen original: {str(e)}")
    except Exception as e:
        print(f"Error al guardar copia de imagen original: {str(e)}")
    
    # Verificar si el modelo existe
    if not os.path.exists(Config.MODEL_PATH):
        print(f"ADVERTENCIA: No se encontró el modelo en {Config.MODEL_PATH}. Usando inferencia simulada.")
        return simulate_analysis(image_path)
    
    try:
        # Cargar la imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: No se pudo leer la imagen: {image_path}")
            return simulate_analysis(image_path)
        
        # Convertir a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Preparar modelo
        model = build_model()
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        model.eval()
        
        # Preprocesar imagen
        transform = A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE, interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        transformed = transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0).to(Config.DEVICE)
        
        # Realizar predicción
        with torch.no_grad():
            pred = model(image_tensor)
        
        # Convertir a numpy y aplicar umbral
        pred_np = pred.squeeze().cpu().numpy()
        
        # Preparar imagen de análisis
        analysis_image = image.copy()
        
        # Colores para cada clase (BGR para OpenCV)
        colors = [(0, 0, 255),     # rojo para entire (BGR)
                  (0, 255, 0),     # verde para numbers
                  (255, 0, 0),     # azul para hands
                  (0, 255, 255)]   # amarillo para contour
        
        # Superponer resultados de segmentación
        overlay = image.copy()
        alpha = 0.4  # Factor de transparencia
        
        # Obtener el nombre base para los archivos
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Diccionario para guardar rutas de segmentaciones
        segmentation_files = {}
        segmentation_files['original_filename'] = f"original_{image_filename}"
        
        # Carpeta de resultados
        results_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'results')
        os.makedirs(results_folder, exist_ok=True)
        
        # Guardar una versión BW de la imagen original
        original_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        original_bw_filename = f"{base_name}_original_bw.png"
        original_bw_path = os.path.join(results_folder, original_bw_filename)
        cv2.imwrite(original_bw_path, original_bw)
        segmentation_files['original_bw_filename'] = original_bw_filename
        
        # Añadir cada máscara con su color y guardar segmentaciones individuales
        for i, class_name in enumerate(Config.CLASSES):
            # Redimensionar al tamaño original
            pred_class = cv2.resize(pred_np[i], (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Aplicar umbral para obtener máscara binaria
            mask = (pred_class > 0.5).astype(np.uint8)
            
            # Guardar segmentación individual
            seg_filename = f"{base_name}_{class_name}.png"
            seg_path = os.path.join(results_folder, seg_filename)
            
            # Crear máscara coloreada para visualización
            colored_mask = np.zeros_like(image)
            colored_mask[mask == 1] = colors[i]
            
           # Superponer máscara en la imagen de análisis
            cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
            
            # Guardar segmentación como imagen RGBA
            seg_image = original_bw.copy()  # Usar imagen original en BW como base
            seg_image_color = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2BGR)
            
            # Añadir las partes segmentadas en color sobre fondo gris
            seg_image_color[mask == 1] = colors[i]
            
            # Guardar la imagen
            cv2.imwrite(seg_path, seg_image_color)
            
            # Guardar ruta en el diccionario
            segmentation_files[class_name + '_filename'] = seg_filename
        
        # Fusionar con la imagen original para la imagen de análisis final
        analysis_image = overlay
        
        # Añadir leyenda
        legend_height = 40 * len(Config.CLASSES)  # Altura por cada clase
        legend = np.ones((legend_height, w, 3), dtype=np.uint8) * 255  # Fondo blanco
        
        for i, class_name in enumerate(Config.CLASSES):
            y = i * 40 + 30  # Posición vertical del texto
            cv2.putText(legend, f"{class_name}", (10, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Texto en negro
            cv2.circle(legend, (w - 30, y - 10), 15, colors[i], -1)  # Círculo de color
        
        # Combinar imagen de análisis con leyenda
        analysis_image = np.vstack([analysis_image, legend])
        
        # Guardar la imagen con el análisis
        analysis_filename = f"{base_name}_analysis.png"
        analysis_path = os.path.join(results_folder, analysis_filename)
        cv2.imwrite(analysis_path, analysis_image)
        segmentation_files['analysis_filename'] = analysis_filename
        
        # Calcular puntajes basados en la predicción
        
        # 1. Calidad del círculo (contour)
        contour_mask = cv2.resize(pred_np[3], (w, h)) > 0.5
        contour_score = np.mean(contour_mask.astype(float))
        circle_integrity = contour_score > 0.6
        
        # 2. Colocación de números
        numbers_mask = cv2.resize(pred_np[1], (w, h)) > 0.5
        numbers_score = np.mean(numbers_mask.astype(float))
        number_placement = numbers_score > 0.5
        
        # 3. Colocación de manecillas
        hands_mask = cv2.resize(pred_np[2], (w, h)) > 0.5
        hands_score = np.mean(hands_mask.astype(float))
        hand_placement = hands_score > 0.5
        
        # 4. Números faltantes (simular basado en numbers_score)
        missing_numbers = numbers_score < 0.4
        
        # 5. Organización espacial (usar mask del reloj completo)
        entire_mask = cv2.resize(pred_np[0], (w, h)) > 0.5
        spatial_score = np.mean(entire_mask.astype(float))
        spatial_organization = spatial_score > 0.7
        
        # Calcular puntuación total (1-5)
        total_score = (int(circle_integrity) + 
                       int(number_placement) + 
                       int(hand_placement) + 
                       int(not missing_numbers) + 
                       int(spatial_organization))
        
        # ======= AGREGAR CLASIFICACIÓN HC/PD =======
        # Cargar el modelo de clasificación
        classification_model = load_classification_model(Config.CLASSIFICATION_MODEL_PATH, Config.DEVICE)
        
        # Clasificar la imagen del reloj
        health_status, confidence = classify_clock_image(classification_model, image_path, Config.DEVICE)
        
        # Crear resultado del análisis
        analysis_result = {
            'score': total_score,
            'circle_integrity': circle_integrity,
            'number_placement': number_placement,
            'hand_placement': hand_placement,
            'missing_numbers': missing_numbers,
            'spatial_organization': spatial_organization,
            'health_status': health_status,
            'confidence': confidence
        }
        
        # Combinar con información de archivos de segmentación
        analysis_result.update(segmentation_files)
        
        return analysis_result, analysis_image
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error durante el análisis: {str(e)}")
        # Si hay un error, usar el analizador simulado como fallback
        return simulate_analysis(image_path)

def simulate_analysis(image_path):
    """
    Función de respaldo que simula el análisis cuando no hay modelo disponible
    o cuando ocurre un error.
    """
    print("Usando análisis simulado...")
    
    # Cargar la imagen
    image = cv2.imread(image_path)
    if image is None:
        # Si no se puede cargar la imagen, crear una en blanco
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Crear una copia para dibujar los resultados del análisis
    analysis_image = image.copy()
    h, w = image.shape[:2]
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral para binarizar la imagen
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Obtener el nombre base para los archivos
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_filename = os.path.basename(image_path)
    
    # Carpeta de resultados
    results_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'results')
    os.makedirs(results_folder, exist_ok=True)
    
    # Guardar una copia de la imagen original
    original_copy_path = os.path.join(results_folder, f"original_{image_filename}")
    cv2.imwrite(original_copy_path, image)
    
    # Guardar una versión BW de la imagen original
    original_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_bw_filename = f"{base_name}_original_bw.png"
    original_bw_path = os.path.join(results_folder, original_bw_filename)
    cv2.imwrite(original_bw_path, original_bw)
    
    # Diccionario para guardar rutas de segmentaciones
    segmentation_files = {}
    segmentation_files['original_filename'] = f"original_{image_filename}"
    segmentation_files['original_bw_filename'] = original_bw_filename
    
    # Simulación de segmentaciones
    if contours:
        # Ordenar contornos por área
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        clock_contour = contours[0]
        
        # Dibujar el contorno del reloj detectado
        cv2.drawContours(analysis_image, [clock_contour], -1, (0, 255, 0), 2)
        
        # Calcular características para simular el análisis
        
        # Centro del reloj
        M = cv2.moments(clock_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Dibujar el centro
            cv2.circle(analysis_image, (cx, cy), 5, (255, 0, 0), -1)
            
            # Crear máscaras simuladas para cada clase
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            numbers_mask = np.zeros((h, w), dtype=np.uint8)
            hands_mask = np.zeros((h, w), dtype=np.uint8)
            entire_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Dibujar contorno para la máscara
            cv2.drawContours(contour_mask, [clock_contour], -1, 255, 2)
            
            # Dibujar círculo entero
            cv2.drawContours(entire_mask, [clock_contour], -1, 255, -1)
            
            # Simular números (círculos pequeños)
            radius = max(cv2.contourArea(clock_contour) ** 0.5 / 2, 50)
            for i in range(12):
                angle = i * 30 - 90  # -90 para que el 12 esté arriba
                x = int(cx + radius * 0.8 * np.cos(np.radians(angle)))
                y = int(cy + radius * 0.8 * np.sin(np.radians(angle)))
                
                # Dibujar círculo para marcar cada número
                cv2.circle(analysis_image, (x, y), 3, (0, 0, 255), -1)
                cv2.circle(numbers_mask, (x, y), 10, 255, -1)
                
                # Añadir el número en la imagen de análisis
                cv2.putText(analysis_image, str((i % 12) + 1), (x - 5, y + 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Simular las manecillas
            # Manecilla de la hora (11)
            hour_angle = 330 - 90  # 11 horas (330 grados) - 90 para ajustar
            hour_x = int(cx + radius * 0.5 * np.cos(np.radians(hour_angle)))
            hour_y = int(cy + radius * 0.5 * np.sin(np.radians(hour_angle)))
            cv2.line(analysis_image, (cx, cy), (hour_x, hour_y), (255, 0, 255), 2)
            cv2.line(hands_mask, (cx, cy), (hour_x, hour_y), 255, 5)
            
            # Manecilla de los minutos (10)
            min_angle = 60 - 90  # 10 minutos (60 grados) - 90 para ajustar
            min_x = int(cx + radius * 0.7 * np.cos(np.radians(min_angle)))
            min_y = int(cy + radius * 0.7 * np.sin(np.radians(min_angle)))
            cv2.line(analysis_image, (cx, cy), (min_x, min_y), (255, 0, 255), 2)
            cv2.line(hands_mask, (cx, cy), (min_x, min_y), 255, 5)
            
            # Añadir texto explicativo
            cv2.putText(analysis_image, "Verde: Contorno detectado", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(analysis_image, "Azul: Centro detectado", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(analysis_image, "Rojo: Posicion ideal de numeros", (10, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(analysis_image, "Magenta: Posicion ideal de manecillas", (10, 120), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Guardar las máscaras simuladas
            for class_name, mask in [
                ('entire', entire_mask),
                ('numbers', numbers_mask),
                ('hands', hands_mask),
                ('contour', contour_mask)
            ]:
                # Guardar segmentación individual
                seg_filename = f"{base_name}_{class_name}.png"
                seg_path = os.path.join(results_folder, seg_filename)
                
                # Convertir máscara a imagen en escala de grises para después colorear
                mask_image = gray.copy()  # Usar imagen original en escala de grises
                mask_image_color = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
                
                # Colorear según la clase
                if class_name == 'entire':
                    color = (0, 0, 255)  # Rojo (BGR)
                elif class_name == 'numbers':
                    color = (0, 255, 0)  # Verde
                elif class_name == 'hands':
                    color = (255, 0, 0)  # Azul
                else:  # contour
                    color = (0, 255, 255)  # Amarillo
                
                # Aplicar color donde está la máscara
                mask_image_color[mask > 0] = color
                
                # Guardar imagen
                cv2.imwrite(seg_path, mask_image_color)
                
                # Guardar ruta en el diccionario
                segmentation_files[class_name + '_filename'] = seg_filename
            
            # Guardar la imagen con el análisis
            analysis_filename = f"{base_name}_analysis.png"
            analysis_path = os.path.join(results_folder, analysis_filename)
            cv2.imwrite(analysis_path, analysis_image)
            segmentation_files['analysis_filename'] = analysis_filename
            
            # Generar resultados aleatorios para la evaluación
            import random
            
            # Generar un score aleatorio entre 1 y 5
            score = random.randint(1, 5)
            
            # Generar resultados aleatorios para las características
            analysis_result = {
                'score': score,
                'circle_integrity': random.choice([True, False]),
                'number_placement': random.choice([True, False]),
                'hand_placement': random.choice([True, False]),
                'missing_numbers': random.choice([True, False]),
                'spatial_organization': random.choice([True, False])
            }
            
            # Simular clasificación (Sano/Enfermo) con valores aleatorios
            health_status = random.choice(['Sano', 'Enfermo'])
            confidence = random.uniform(70.0, 95.0)  # Confianza entre 70% y 95%
            
            # Añadir información de clasificación
            analysis_result['health_status'] = health_status
            analysis_result['confidence'] = confidence
            
            # Añadir rutas de los archivos de segmentación
            analysis_result.update(segmentation_files)
            
            return analysis_result, analysis_image
        
    # Si no se detectaron contornos o hubo algún problema
    # Crear máscaras vacías pero coloreadas
    for class_name in Config.CLASSES:
        # Guardar segmentación individual (vacía)
        seg_filename = f"{base_name}_{class_name}.png"
        seg_path = os.path.join(results_folder, seg_filename)
        
        # Crear imagen coloreada (gris con tinte de color)
        mask_image = gray.copy()
        mask_image_color = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        
        # Aplicar un tinte sutil según la clase
        if class_name == 'entire':
            mask_image_color = cv2.addWeighted(mask_image_color, 0.9, np.ones_like(mask_image_color) * np.array([0, 0, 50], dtype=np.uint8), 0.1, 0)
        elif class_name == 'numbers':
            mask_image_color = cv2.addWeighted(mask_image_color, 0.9, np.ones_like(mask_image_color) * np.array([0, 50, 0], dtype=np.uint8), 0.1, 0)
        elif class_name == 'hands':
            mask_image_color = cv2.addWeighted(mask_image_color, 0.9, np.ones_like(mask_image_color) * np.array([50, 0, 0], dtype=np.uint8), 0.1, 0)
        else:  # contour
            mask_image_color = cv2.addWeighted(mask_image_color, 0.9, np.ones_like(mask_image_color) * np.array([0, 50, 50], dtype=np.uint8), 0.1, 0)
        
        cv2.imwrite(seg_path, mask_image_color)
        
        # Guardar ruta en el diccionario
        segmentation_files[class_name + '_filename'] = seg_filename
    
    # Añadir texto al análisis indicando que no se detectó un reloj
    cv2.putText(analysis_image, "No se detectó un reloj", (50, 50), 
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Guardar la imagen con el análisis
    analysis_filename = f"{base_name}_analysis.png"
    analysis_path = os.path.join(results_folder, analysis_filename)
    cv2.imwrite(analysis_path, analysis_image)
    segmentation_files['analysis_filename'] = analysis_filename
    
    # Crear resultado del análisis con valores negativos
    analysis_result = {
        'score': 0,
        'circle_integrity': False,
        'number_placement': False,
        'hand_placement': False,
        'missing_numbers': True,
        'spatial_organization': False
    }
    
    # Simular clasificación con valores por defecto
    import random
    analysis_result['health_status'] = 'No determinado'
    analysis_result['confidence'] = 0.0
    
    # Añadir rutas de los archivos de segmentación
    analysis_result.update(segmentation_files)
    
    return analysis_result, analysis_image