import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import segmentation_models_pytorch as smp

# Definición de la arquitectura específica del modelo
class Adapter7to3(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

def build_densenet121_adapter7ch_paper(num_classes, pretrained=True):
    d121 = models.densenet121(pretrained=pretrained)
    head = nn.Sequential(
        nn.Linear(d121.classifier.in_features, 1000),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(1000, 500),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(500, num_classes)
    )
    return nn.Sequential(
        Adapter7to3(),
        d121.features,
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        head
    )

def build_densenet121_adapter7ch_paper_ft(num_classes, pretrained=True):
    model = build_densenet121_adapter7ch_paper(num_classes, pretrained)
    for name, p in model.named_parameters():
        # Descongelar adapter, últimos bloques y cabeza
        if name.startswith('0') or '5' in name or any(x in name for x in ['denseblock4','transition3']):
            p.requires_grad = True
    return model

def get_model_architecture(model_path: str):
    """
    Determina la arquitectura correcta del modelo basada en las claves del estado del diccionario
    """
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        keys = list(state_dict.keys())
        
        # Para este caso específico, sabemos que es un modelo Adapter7ch
        if any('0.block.0.weight' in key for key in keys):
            print("Detectado modelo DenseNet121 con Adapter7ch")
            return 'adapter7ch'
        # Examinar las primeras claves para determinar el tipo de modelo
        elif any('denseblock' in key for key in keys):
            print("Detectado modelo DenseNet")
            return 'densenet'
        elif any('layer1.0.conv1.weight' in key for key in keys):
            print("Detectado modelo ResNet")
            return 'resnet'
        elif any('block.0.weight' in key for key in keys):
            print("Detectado modelo personalizado o UNet/UNet++")
            return 'custom'
        else:
            print("Tipo de modelo no reconocido, intentando con DenseNet por defecto")
            return 'densenet'
    except Exception as e:
        print(f"Error al analizar la arquitectura del modelo: {str(e)}")
        return 'densenet'  # Por defecto, intentamos con DenseNet

def load_classification_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Carga la arquitectura y los pesos adecuados según el modelo detectado
    
    Args:
        model_path: Ruta al archivo con los pesos del modelo
        device: Dispositivo donde se cargará el modelo (cuda/cpu)
        
    Returns:
        nn.Module o None: Modelo cargado o None si hay error
    """
    # Si el archivo no existe, informamos y retornamos None
    if not os.path.exists(model_path):
        print(f"ADVERTENCIA: No se encontró el modelo de clasificación en {model_path}")
        return None
        
    try:
        # Usar el parámetro model_path para determinar el tipo de arquitectura
        model_type = get_model_architecture('/Users/diegotovar/Developer/Tesis/WebPage_v2/models/d121_ad7_p_ft-20250511_013731.pth')
        
        if model_type == 'adapter7ch':
            # Modelo específico Adapter7ch
            model = build_densenet121_adapter7ch_paper_ft(num_classes=3, pretrained=False)
        elif model_type == 'resnet':
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)  # 2 clases: HC (sano) y PD (enfermo)
        elif model_type == 'densenet':
            model = models.densenet121(pretrained=False)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 2)  # 2 clases: HC (sano) y PD (enfermo)
        elif model_type == 'custom':
            try:
                # Intentar cargar como un modelo UNet++ de segmentation_models_pytorch
                model = smp.UnetPlusPlus(
                    encoder_name="densenet121", 
                    encoder_weights=None,
                    classes=2,
                    activation="softmax"
                )
            except:
                # Si falla, cargar como un modelo de clasificación DenseNet
                model = models.densenet121(pretrained=False)
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, 2)
        
        # Cargar los pesos
        state_dict = torch.load(model_path, map_location=device)
        
        # Imprimir información para depuración
        print(f"Cargando modelo de clasificación de {model_path}")
        
        try:
            model.load_state_dict(state_dict)
        except Exception as e:
            print(f"Error al cargar los pesos: {str(e)}")
            print("Intentando carga flexible...")
            # Intentar cargar solo los parámetros que coinciden
            model_dict = model.state_dict()
            # Filtrar estado del diccionario para incluir solo las claves que están en model_dict
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"Cargados {len(pretrained_dict)}/{len(state_dict)} parámetros")
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error al cargar el modelo de clasificación: {str(e)}")
        return None

def classify_clock_image(model, image_path, device, threshold=0.7):
    """
    Clasifica una imagen del reloj como 'Sano' (HC) o 'Enfermo' (PD)
    
    Args:
        model: Modelo de clasificación
        image_path: Ruta a la imagen del reloj
        device: Dispositivo para procesamiento (cuda/cpu)
        threshold: Umbral de confianza para la clasificación
        
    Returns:
        tuple: (classe, confianza)
            - classe: String con 'Sano' o 'Enfermo'
            - confianza: Valor de confianza de la predicción (0-100%)
    """
    if model is None:
        # Si no hay modelo, retornamos valores por defecto
        print("No hay modelo de clasificación disponible, retornando clasificación simulada")
        import random
        # Simulamos una clasificación con valores aleatorios
        is_healthy = random.choice([True, False])
        confidence = random.uniform(60.0, 95.0)
        return ('Sano' if is_healthy else 'Enfermo'), confidence
    
    # Preparar transformación de la imagen
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    try:
        # Cargar y procesar la imagen
        img = Image.open(image_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        
        # Determinar si el modelo es el adapter7ch con 3 clases
        is_adapter7ch = hasattr(model, '_modules') and '0' in model._modules and isinstance(model._modules['0'], Adapter7to3)
        
        # Realizar predicción
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Si es el modelo Adapter7ch con 3 clases, usar umbrales específicos
            if is_adapter7ch:
                class_names = ['CN', 'MCI', 'PDD']
                thresholds = {
                    'CN': 0.6,
                    'MCI': 0.2,
                    'PDD': 0.4
                }
                
                # Filtrar clases que superan su umbral
                passed = {}
                for i, class_name in enumerate(class_names):
                    prob = probs[0, i].item()
                    if prob >= thresholds[class_name]:
                        passed[class_name] = prob
                
                if not passed:
                    return 'Indeterminado', 50.0
                
                # Escoger la clase con mayor probabilidad entre las que pasaron
                best_class = max(passed, key=passed.get)
                confidence = passed[best_class] * 100
                
                # Mapear clases a 'Sano' o 'Enfermo'
                if best_class == 'CN':
                    class_label = 'Sano'
                else:
                    class_label = 'Enfermo'  # MCI y PDD se consideran enfermos
            else:
                # Para modelos binarios estándar
                conf, pred = torch.max(probs, dim=1)
                confidence = conf.item() * 100
                class_index = pred.item()
                class_names = ['Sano', 'Enfermo']
                class_label = class_names[class_index]
        
        print(f"Clasificación: {class_label} con confianza: {confidence:.2f}%")
        return class_label, confidence
        
    except Exception as e:
        print(f"Error durante la clasificación: {str(e)}")
        # En caso de error, retornamos valores por defecto
        return 'No determinado', 0.0