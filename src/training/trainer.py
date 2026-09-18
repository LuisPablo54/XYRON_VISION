# Entrenamiento del modelo XYRON Vision

# Librerias necesarias
import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.class_weights import calcular_peso_clase, recolectar_etiquetas
from src.data.dataset import DatasetSoldadura
from src.models.xyron_model import XyronMobileNetV2
from src.training.checkpoint_utils import ParoTemprano, guardar_modelo


def obtener_dispositivo():
    """Devuelvo la GPU cuando CUDA esta disponible y la CPU en caso contrario."""
    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')


def crear_cargadores():
    """Armo los cargadores de entrenamiento y validacion, y devuelvo tambien el dataset de entrenamiento para los pesos de clase."""
    dataset_entrenamiento = DatasetSoldadura(
        os.path.join(RUTA_DATOS, 'train'), NUM_CLASES, get_train_transforms(IMG_SIZE)
    )
    dataset_validacion = DatasetSoldadura(
        os.path.join(RUTA_DATOS, 'valid'), NUM_CLASES, get_val_transforms(IMG_SIZE)
    )

    cargador_entrenamiento = DataLoader(
        dataset_entrenamiento,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    cargador_validacion = DataLoader(
        dataset_validacion,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    return cargador_entrenamiento, cargador_validacion, dataset_entrenamiento


def calcular_perdida(salida_clase, salida_caja, vector_clases, caja, mascara, criterio_clase):
    """Sumo la entropia binaria de las clases y el error de la caja, ignorando las imagenes que quedaron sin caja visible."""
    perdida_clase = criterio_clase(salida_clase, vector_clases)
    error_caja = nn.functional.smooth_l1_loss(salida_caja, caja, reduction='none').mean(dim=1, keepdim=True)
    perdida_caja = (error_caja * mascara).sum() / mascara.sum().clamp(min=1.0)

    return perdida_clase + PESO_CAJA * perdida_caja


def yolo_a_esquinas(cajas):
    """Paso las cajas de centro y tamano a esquinas para poder intersecarlas."""
    mitad_ancho = cajas[:, 2] / 2
    mitad_alto = cajas[:, 3] / 2

    return torch.stack([
        cajas[:, 0] - mitad_ancho,
        cajas[:, 1] - mitad_alto,
        cajas[:, 0] + mitad_ancho,
        cajas[:, 1] + mitad_alto,
    ], dim=1)


def iou_cajas(predichas, reales):
    """Calculo la interseccion sobre union de cada par de cajas predicha y real."""
    caja_a = yolo_a_esquinas(predichas)
    caja_b = yolo_a_esquinas(reales)

    ancho = (torch.minimum(caja_a[:, 2], caja_b[:, 2]) - torch.maximum(caja_a[:, 0], caja_b[:, 0])).clamp(min=0)
    alto = (torch.minimum(caja_a[:, 3], caja_b[:, 3]) - torch.maximum(caja_a[:, 1], caja_b[:, 1])).clamp(min=0)

    interseccion = ancho * alto
    union = predichas[:, 2] * predichas[:, 3] + reales[:, 2] * reales[:, 3] - interseccion

    return interseccion / union.clamp(min=1e-6)


def entrenar_epoca(modelo, cargador, optimizador, criterio_clase, escalador, dispositivo):
    """Recorro una epoca completa de entrenamiento y devuelvo la perdida promedio por imagen."""
    modelo.train()
    perdida_acumulada = 0.0
    muestras = 0

    for imagenes, vector_clases, cajas, mascaras in cargador:
        imagenes = imagenes.to(dispositivo, non_blocking=True)
        vector_clases = vector_clases.to(dispositivo, non_blocking=True)
        cajas = cajas.to(dispositivo, non_blocking=True)
        mascaras = mascaras.to(dispositivo, non_blocking=True)

        optimizador.zero_grad(set_to_none=True)

        with torch.autocast(device_type=dispositivo.type, enabled=dispositivo.type == 'cuda'):
            salida_clase, salida_caja = modelo(imagenes)
            perdida = calcular_perdida(salida_clase, salida_caja, vector_clases, cajas, mascaras, criterio_clase)

        escalador.scale(perdida).backward()
        escalador.step(optimizador)
        escalador.update()

        perdida_acumulada += perdida.item() * imagenes.size(0)
        muestras += imagenes.size(0)

    return perdida_acumulada / max(muestras, 1)


@torch.no_grad()
def validar_epoca(modelo, cargador, criterio_clase, dispositivo):
    """Evaluo la validacion y devuelvo la perdida, el F1 macro de las clases y la IoU media de la caja principal."""
    modelo.eval()
    perdida_acumulada = 0.0
    muestras = 0

    verdaderos_positivos = torch.zeros(NUM_CLASES, device=dispositivo)
    falsos_positivos = torch.zeros(NUM_CLASES, device=dispositivo)
    falsos_negativos = torch.zeros(NUM_CLASES, device=dispositivo)
    iou_acumulada = 0.0
    cajas_validas = 0

    for imagenes, vector_clases, cajas, mascaras in cargador:
        imagenes = imagenes.to(dispositivo, non_blocking=True)
        vector_clases = vector_clases.to(dispositivo, non_blocking=True)
        cajas = cajas.to(dispositivo, non_blocking=True)
        mascaras = mascaras.to(dispositivo, non_blocking=True)

        with torch.autocast(device_type=dispositivo.type, enabled=dispositivo.type == 'cuda'):
            salida_clase, salida_caja = modelo(imagenes)
            perdida = calcular_perdida(salida_clase, salida_caja, vector_clases, cajas, mascaras, criterio_clase)

        perdida_acumulada += perdida.item() * imagenes.size(0)
        muestras += imagenes.size(0)

        prediccion = (torch.sigmoid(salida_clase.float()) >= UMBRAL_CLASE).float()
        verdaderos_positivos += (prediccion * vector_clases).sum(dim=0)
        falsos_positivos += (prediccion * (1 - vector_clases)).sum(dim=0)
        falsos_negativos += ((1 - prediccion) * vector_clases).sum(dim=0)

        indices = mascaras.squeeze(1) > 0
        if indices.any():
            iou_acumulada += iou_cajas(salida_caja.float()[indices], cajas[indices]).sum().item()
            cajas_validas += int(indices.sum().item())

    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos).clamp(min=1e-6)
    sensibilidad = verdaderos_positivos / (verdaderos_positivos + falsos_negativos).clamp(min=1e-6)
    f1_macro = (2 * precision * sensibilidad / (precision + sensibilidad).clamp(min=1e-6)).mean().item()

    return perdida_acumulada / max(muestras, 1), f1_macro, iou_acumulada / max(cajas_validas, 1)


def entrenar():
    """Ejecuto el entrenamiento completo, guardo el mejor punto de control y corto cuando la validacion deja de mejorar."""
    dispositivo = obtener_dispositivo()
    nombre_gpu = torch.cuda.get_device_name(0) if dispositivo.type == 'cuda' else 'sin GPU'
    print(f"Dispositivo: {dispositivo} | {nombre_gpu}")

    torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.dirname(RUTA_CHECKPOINT), exist_ok=True)

    cargador_entrenamiento, cargador_validacion, dataset_entrenamiento = crear_cargadores()
    peso_clase = calcular_peso_clase(recolectar_etiquetas(dataset_entrenamiento), NUM_CLASES).to(dispositivo)

    modelo = XyronMobileNetV2(NUM_CLASES).to(dispositivo)
    criterio_clase = nn.BCEWithLogitsLoss(pos_weight=peso_clase)
    parametros_entrenables = [parametro for parametro in modelo.parameters() if parametro.requires_grad]
    optimizador = torch.optim.Adam(parametros_entrenables, lr=TASA_APRENDIZAJE)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode='min', factor=0.5, patience=2)
    escalador = torch.amp.GradScaler(device=dispositivo.type, enabled=dispositivo.type == 'cuda')
    paro_temprano = ParoTemprano(PACIENCIA)

    for epoca in range(1, EPOCAS + 1):
        perdida_entrenamiento = entrenar_epoca(
            modelo, cargador_entrenamiento, optimizador, criterio_clase, escalador, dispositivo
        )
        perdida_validacion, f1_macro, iou_media = validar_epoca(
            modelo, cargador_validacion, criterio_clase, dispositivo
        )
        planificador.step(perdida_validacion)

        print(
            f"Epoca {epoca}/{EPOCAS} | perdida train {perdida_entrenamiento:.4f} | "
            f"perdida val {perdida_validacion:.4f} | F1 macro {f1_macro:.4f} | IoU {iou_media:.4f}"
        )

        metricas = {'perdida_validacion': perdida_validacion, 'f1_macro': f1_macro, 'iou_media': iou_media}

        if paro_temprano.actualizar(perdida_validacion):
            guardar_modelo(modelo, optimizador, epoca, metricas, RUTA_CHECKPOINT)
            print(f"Modelo guardado en {RUTA_CHECKPOINT}")

        if paro_temprano.detener:
            print(f"Paro temprano en la epoca {epoca}")
            break

    return modelo


# Parametros de configuracion
RUTA_DATOS = 'data_1'
RUTA_CHECKPOINT = os.path.join('checkpoints', 'xyron_mnv2.pt')
IMG_SIZE = 640
NUM_CLASES = 3
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCAS = 30
TASA_APRENDIZAJE = 1e-3
PESO_CAJA = 5.0
UMBRAL_CLASE = 0.5
PACIENCIA = 5

if __name__ == "__main__":
    entrenar()
