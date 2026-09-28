# Archivo para generar aumentacion de datos

# Importamos librerias necesarias
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Función principal
def get_train_transforms(img_size):
    """Devuelvo las transformaciones de entrenamiento y entrego la imagen ya normalizada como tensor de PyTorch."""
    transforms = A.Compose([
        A.HorizontalFlip(p=0.5), # 50% posibilidades de voltear horizontalmente la imagen
        A.RandomBrightnessContrast(p=0.3), # Posibiliad de que cambie de exposición entre +- 20% 
        A.Rotate(limit=15, p=0.3),  # Posibilidad de que rote con un limiete de 15 grados
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEDIA_IMAGENET, std=DESVIACION_IMAGENET),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['clases'], clip=True, min_visibility=0.3))

    return transforms


def get_val_transforms(img_size):
    # Devuelvo las transformaciones de validacion y prueba, sin aleatoriedad para que las metricas sean reproducibles
    transforms = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEDIA_IMAGENET, std=DESVIACION_IMAGENET),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['clases'], clip=True, min_visibility=0.3))

    return transforms


# Estadisticas de ImageNet, las mismas con las que se entreno MobileNetV2
MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESVIACION_IMAGENET = (0.229, 0.224, 0.225)
