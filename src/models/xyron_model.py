# Arquitectura CNN + MobileNetV2

# Librerias necesarias

## Modelo CNN
import math

import torch
from torch import nn
from torchvision.ops import batched_nms

# Modelo MobileNetV2
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class XyronMobileNetV2(nn.Module):
    """Armo el extractor MobileNetV2 con dos cabezas de salida: clasificacion multietiqueta de la imagen y una cabeza densa que detecta todas las cajas sobre un FPN de pasos 8, 16 y 32."""

    def __init__(self, num_clases, descongelar_desde=None):
        super().__init__()
        self.num_clases = num_clases

        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.extractor = base_model.features # cabeza original de ImageNet cortada

        for parametro in self.extractor.parameters():
            parametro.requires_grad = False # Se congela

        # Los ultimos bloques se pueden descongelar para que el extractor se adapte a las soldaduras
        self.descongelar_desde = len(self.extractor) if descongelar_desde is None else descongelar_desde
        for parametro in self.extractor[self.descongelar_desde:].parameters():
            parametro.requires_grad = True

        # Capas convolucionales propias sobre los mapas de características de MobileNet
        self.convoluciones = nn.Sequential(
            nn.Conv2d(CANALES_MNV2, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, CANALES_FPN, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(CANALES_FPN),
        )

        self.agrupamiento = nn.AdaptiveAvgPool2d(1)

        # Entrega logits porque la perdida BCEWithLogitsLoss aplica la sigmoide internamente
        self.cabeza_clase = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(CANALES_FPN, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_clases),
        )

        # FPN: las conexiones laterales llevan los mapas de paso 8 y 16 a los mismos canales que el de paso 32
        self.lateral_p3 = nn.Conv2d(CANALES_C3, CANALES_FPN, kernel_size=1)
        self.lateral_p4 = nn.Conv2d(CANALES_C4, CANALES_FPN, kernel_size=1)
        self.suavizado_p3 = nn.Conv2d(CANALES_FPN, CANALES_FPN, kernel_size=3, padding='same')
        self.suavizado_p4 = nn.Conv2d(CANALES_FPN, CANALES_FPN, kernel_size=3, padding='same')

        # Cabeza densa compartida por los tres niveles; GroupNorm porque mezcla mapas de escalas distintas
        self.cabeza_densa = nn.Sequential(
            nn.Conv2d(CANALES_FPN, CANALES_FPN, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.GroupNorm(32, CANALES_FPN),
            nn.Conv2d(CANALES_FPN, CANALES_FPN, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.GroupNorm(32, CANALES_FPN),
        )

        # Un mapa de confianza por clase; el sesgo inicial hace que todas las celdas arranquen con probabilidad baja
        self.salida_confianza = nn.Conv2d(CANALES_FPN, num_clases, kernel_size=1)
        nn.init.constant_(self.salida_confianza.bias, -math.log((1 - PROBABILIDAD_INICIAL) / PROBABILIDAD_INICIAL))

        # Centralidad de FCOS: baja el puntaje de las celdas del borde, que predicen cajas de peor calidad
        self.salida_centralidad = nn.Conv2d(CANALES_FPN, num_clases, kernel_size=1)

        # Cuatro distancias (izquierda, arriba, derecha, abajo) por clase, para que un Defect dentro de un Bad Weld tenga su propia caja
        self.salida_caja = nn.Conv2d(CANALES_FPN, num_clases * 4, kernel_size=1)

    def train(self, modo=True):
        """Mantengo en modo evaluacion la parte congelada del extractor para que sus BatchNorm no cambien las estadisticas de ImageNet."""
        super().train(modo)
        self.extractor[:self.descongelar_desde].eval()

        return self

    def forward(self, imagenes):
        """Devuelvo los logits de la imagen y las salidas de todas las celdas de los tres niveles aplanadas: confianza y centralidad [lote, clases, puntos] y caja [lote, clases, 4, puntos]."""
        c3 = self.extractor[:CORTE_C3](imagenes)
        c4 = self.extractor[CORTE_C3:CORTE_C4](c3)
        p5 = self.convoluciones(self.extractor[CORTE_C4:](c4))

        salida_clase = self.cabeza_clase(torch.flatten(self.agrupamiento(p5), 1))

        # Camino descendente del FPN: el contexto de p5 se suma a los mapas finos que conservan las cajas pequenas
        p4 = self.suavizado_p4(self.lateral_p4(c4) + nn.functional.interpolate(p5, size=c4.shape[-2:], mode='nearest'))
        p3 = self.suavizado_p3(self.lateral_p3(c3) + nn.functional.interpolate(p4, size=c3.shape[-2:], mode='nearest'))

        confianzas = []
        centralidades = []
        cajas = []
        for nivel in (p3, p4, p5):
            densas = self.cabeza_densa(nivel)
            confianzas.append(self.salida_confianza(densas).flatten(2))
            centralidades.append(self.salida_centralidad(densas).flatten(2))
            cajas.append(self.salida_caja(densas).view(densas.shape[0], self.num_clases, 4, -1))

        return salida_clase, torch.cat(confianzas, dim=2), torch.cat(centralidades, dim=2), torch.cat(cajas, dim=3)


def generar_puntos(tamano_imagen, dispositivo=None):
    """Devuelvo los centros normalizados de todas las celdas de los tres niveles en el mismo orden que la salida del modelo, el tamano de celda de cada una y su nivel."""
    centros = []
    tamanos_celda = []
    niveles = []

    for nivel, paso in enumerate(PASOS_FPN):
        malla = tamano_imagen // paso
        coordenadas = (torch.arange(malla, device=dispositivo) + 0.5) / malla
        centros_y, centros_x = torch.meshgrid(coordenadas, coordenadas, indexing='ij')
        centros.append(torch.stack([centros_x.flatten(), centros_y.flatten()], dim=1))
        tamanos_celda.append(torch.full((malla * malla,), 1.0 / malla, device=dispositivo))
        niveles.append(torch.full((malla * malla,), nivel, dtype=torch.long, device=dispositivo))

    return torch.cat(centros), torch.cat(tamanos_celda), torch.cat(niveles)


def decodificar_cajas(salida_caja, tamano_imagen):
    """Convierto las distancias predichas (en celdas de su nivel) a cajas en esquinas normalizadas (x1, y1, x2, y2) con forma [lote, clases, 4, puntos]."""
    centros, tamanos_celda, _ = generar_puntos(tamano_imagen, salida_caja.device)
    distancias = torch.exp(salida_caja.float().clamp(max=LIMITE_EXPONENTE)) * tamanos_celda

    return torch.stack([
        centros[:, 0] - distancias[:, :, 0],
        centros[:, 1] - distancias[:, :, 1],
        centros[:, 0] + distancias[:, :, 2],
        centros[:, 1] + distancias[:, :, 3],
    ], dim=2)


@torch.no_grad()
def postprocesar(salida_confianza, salida_centralidad, salida_caja, tamano_imagen, umbral_confianza, umbral_iou, max_detecciones=100):
    """Combino confianza y centralidad, filtro por umbral y aplico NMS por grupo: Bad Weld y Good Weld compiten entre si porque una soldadura no puede ser ambas, y Defect va aparte para no suprimirlo dentro de un Bad Weld; devuelvo una lista de detecciones por imagen."""
    puntajes_todos = torch.sqrt(torch.sigmoid(salida_confianza.float()) * torch.sigmoid(salida_centralidad.float()))
    cajas = decodificar_cajas(salida_caja, tamano_imagen).clamp(0, 1)
    num_clases, total_puntos = puntajes_todos.shape[1:]
    grupos = torch.tensor(GRUPOS_NMS, device=puntajes_todos.device)
    detecciones = []

    for indice in range(puntajes_todos.shape[0]):
        puntajes = puntajes_todos[indice].reshape(-1)
        cajas_imagen = cajas[indice].permute(0, 2, 1).reshape(-1, 4)
        clases = torch.arange(num_clases, device=puntajes.device).repeat_interleave(total_puntos)

        seleccion = torch.nonzero(puntajes >= umbral_confianza).flatten()
        seleccion = seleccion[puntajes[seleccion].argsort(descending=True)[:MAX_CANDIDATOS_NMS]]
        puntajes, cajas_imagen, clases = puntajes[seleccion], cajas_imagen[seleccion], clases[seleccion]

        conservadas = batched_nms(cajas_imagen, puntajes, grupos[clases], umbral_iou)[:max_detecciones]
        detecciones.append({
            'cajas': cajas_imagen[conservadas],
            'puntajes': puntajes[conservadas],
            'clases': clases[conservadas],
        })

    return detecciones


# Size de la imagenes (EDA lo mostro)
IMG_SIZE_MNV2 = 640

# Canales que entrega el ultimo bloque de MobileNetV2 y los de las salidas de paso 8 (features[6]) y paso 16 (features[13])
CANALES_MNV2 = 1280
CANALES_C3 = 32
CANALES_C4 = 96

# Indices donde se corta MobileNetV2 para sacar los mapas de paso 8 y 16
CORTE_C3 = 7
CORTE_C4 = 14

# Pasos de los niveles del FPN y rango de distancia maxima al borde (en pixeles) que atiende cada uno
PASOS_FPN = (8, 16, 32)
RANGOS_ESCALA = ((0.0, 64.0), (64.0, 128.0), (128.0, float('inf')))
CANALES_FPN = 128

# Clases de data.yaml: Bad Weld, Good Weld, Defect
NUM_CLASES = 3

# Probabilidad con la que arranca cada celda, casi todas son fondo (inicializacion de RetinaNet)
PROBABILIDAD_INICIAL = 0.01

# Tope del exponente de las distancias: exp(4) equivale a unas 54 celdas de su nivel
LIMITE_EXPONENTE = 4.0

# Candidatos con mayor puntaje que pasan al NMS por imagen
MAX_CANDIDATOS_NMS = 1000

# Grupo de NMS de cada clase (Bad Weld, Good Weld, Defect): las dos primeras se suprimen entre si
GRUPOS_NMS = (0, 0, 1)

if __name__ == "__main__":
    modelo = XyronMobileNetV2(NUM_CLASES, descongelar_desde=14)
    entrenables = sum(parametro.numel() for parametro in modelo.parameters() if parametro.requires_grad)
    congelados = sum(parametro.numel() for parametro in modelo.parameters() if not parametro.requires_grad)

    print(modelo)
    print(f"Parametros entrenables: {entrenables:,}")
    print(f"Parametros congelados: {congelados:,}")

    modelo.eval()
    salida_clase, salida_confianza, salida_centralidad, salida_caja = modelo(
        torch.zeros(1, 3, IMG_SIZE_MNV2, IMG_SIZE_MNV2)
    )
    print(
        f"Salida clase: {tuple(salida_clase.shape)} | Salida confianza: {tuple(salida_confianza.shape)} | "
        f"Salida centralidad: {tuple(salida_centralidad.shape)} | Salida caja: {tuple(salida_caja.shape)}"
    )
