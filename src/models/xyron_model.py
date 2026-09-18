# Arquitectura CNN + MobileNetV2

# Librerias necesarias

## Modelo CNN
import torch
from torch import nn

# Modelo MobileNetV2
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class XyronMobileNetV2(nn.Module):
    """Armo el extractor MobileNetV2 congelado con dos cabezas de salida: clasificacion multietiqueta de defectos y regresion de la caja principal."""

    def __init__(self, num_clases):
        super().__init__()

        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.extractor = base_model.features # cabeza original de ImageNet cortada

        for parametro in self.extractor.parameters():
            parametro.requires_grad = False # Se congela

        # Capas convolucionales propias sobre los mapas de características de MobileNet
        self.convoluciones = nn.Sequential(
            nn.Conv2d(CANALES_MNV2, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        self.agrupamiento = nn.AdaptiveAvgPool2d(1)

        # Entrega logits porque la perdida BCEWithLogitsLoss aplica la sigmoide internamente
        self.cabeza_clase = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_clases),
        )

        # Entrega la caja en formato yolo normalizado, por eso la sigmoide al final
        self.cabeza_caja = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Sigmoid(),
        )

    def forward(self, imagenes):
        caracteristicas = self.extractor(imagenes)
        caracteristicas = self.convoluciones(caracteristicas)
        caracteristicas = torch.flatten(self.agrupamiento(caracteristicas), 1)

        return self.cabeza_clase(caracteristicas), self.cabeza_caja(caracteristicas)


# Size de la imagenes (EDA lo mostro)
IMG_SIZE_MNV2 = 640

# Canales que entrega el ultimo bloque de MobileNetV2
CANALES_MNV2 = 1280

# Clases de data.yaml: Bad Weld, Good Weld, Defect
NUM_CLASES = 3

if __name__ == "__main__":
    modelo = XyronMobileNetV2(NUM_CLASES)
    entrenables = sum(parametro.numel() for parametro in modelo.parameters() if parametro.requires_grad)
    congelados = sum(parametro.numel() for parametro in modelo.parameters() if not parametro.requires_grad)

    print(modelo)
    print(f"Parametros entrenables: {entrenables:,}")
    print(f"Parametros congelados: {congelados:,}")

    salida_clase, salida_caja = modelo(torch.zeros(1, 3, IMG_SIZE_MNV2, IMG_SIZE_MNV2))
    print(f"Salida clase: {tuple(salida_clase.shape)} | Salida caja: {tuple(salida_caja.shape)}")
