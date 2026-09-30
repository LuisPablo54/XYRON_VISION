# Entrenamiento del modelo XYRON Vision

# Librerias necesarias
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.ops import complete_box_iou_loss, sigmoid_focal_loss

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.class_weights import calcular_peso_clase, recolectar_etiquetas
from src.data.dataset import DatasetSoldadura, agrupar_lote
from src.evaluation.metricas_deteccion import evaluar_detecciones
from src.models.xyron_model import XyronMobileNetV2, decodificar_cajas, generar_puntos, postprocesar
from src.training.checkpoint_utils import ParoTemprano, guardar_modelo


def obtener_dispositivo():
    """Devuelvo la GPU cuando CUDA esta disponible y la CPU en caso contrario."""
    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')


def crear_cargadores():
    """Armo los cargadores de entrenamiento y validacion, y devuelvo tambien el dataset de entrenamiento para los pesos de clase."""
    dataset_entrenamiento = DatasetSoldadura(
        os.path.join(RUTA_DATOS, 'train'), NUM_CLASES, get_train_transforms(IMG_SIZE), IMG_SIZE
    )
    dataset_validacion = DatasetSoldadura(
        os.path.join(RUTA_DATOS, 'valid'), NUM_CLASES, get_val_transforms(IMG_SIZE), IMG_SIZE
    )

    cargador_entrenamiento = DataLoader(
        dataset_entrenamiento,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=agrupar_lote,
    )
    cargador_validacion = DataLoader(
        dataset_validacion,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        collate_fn=agrupar_lote,
    )

    return cargador_entrenamiento, cargador_validacion, dataset_entrenamiento


def calcular_centralidad(cajas, centros):
    """Calculo la centralidad de FCOS: vale 1 en el centro de la caja y cae hacia 0 en los bordes."""
    izquierda = centros[:, 0] - cajas[:, 0]
    arriba = centros[:, 1] - cajas[:, 1]
    derecha = cajas[:, 2] - centros[:, 0]
    abajo = cajas[:, 3] - centros[:, 1]

    horizontal = torch.minimum(izquierda, derecha) / torch.maximum(izquierda, derecha).clamp(min=1e-6)
    vertical = torch.minimum(arriba, abajo) / torch.maximum(arriba, abajo).clamp(min=1e-6)

    return torch.sqrt((horizontal * vertical).clamp(min=0))


def calcular_perdida(salidas, vector_clases, mapa_confianza, mapa_cajas, criterio_clase):
    """Sumo la entropia binaria de la imagen, la focal loss de las celdas, la centralidad y L1 + CIoU de las cajas; estas dos ultimas solo en las celdas positivas."""
    salida_clase, salida_confianza, salida_centralidad, salida_caja = salidas
    perdida_clase = criterio_clase(salida_clase.float(), vector_clases)

    # La focal loss baja el peso de las celdas de fondo faciles, que son casi todas
    positivos = mapa_confianza > 0
    num_positivos = positivos.sum().clamp(min=1)
    perdida_confianza = sigmoid_focal_loss(
        salida_confianza.float(), mapa_confianza, alpha=ALFA_FOCAL, gamma=GAMMA_FOCAL, reduction='sum'
    ) / num_positivos

    centros, _, _ = generar_puntos(IMG_SIZE, salida_caja.device)
    cajas_predichas = decodificar_cajas(salida_caja, IMG_SIZE).permute(0, 1, 3, 2)[positivos]
    cajas_reales = mapa_cajas.permute(0, 1, 3, 2)[positivos]
    centros_positivos = centros.expand(*positivos.shape, 2)[positivos]

    perdida_centralidad = torch.zeros((), device=salida_caja.device)
    perdida_l1 = torch.zeros((), device=salida_caja.device)
    perdida_ciou = torch.zeros((), device=salida_caja.device)
    if len(cajas_reales) > 0:
        centralidad_real = calcular_centralidad(cajas_reales, centros_positivos)
        perdida_centralidad = nn.functional.binary_cross_entropy_with_logits(
            salida_centralidad.float()[positivos], centralidad_real
        )

        # Como en FCOS, las celdas cercanas al centro pesan mas en la regresion que las del borde
        peso_celda = centralidad_real / centralidad_real.sum().clamp(min=1e-6)
        error_l1 = nn.functional.l1_loss(cajas_predichas, cajas_reales, reduction='none').sum(dim=1)
        perdida_l1 = (error_l1 * peso_celda).sum()
        perdida_ciou = (complete_box_iou_loss(cajas_predichas, cajas_reales, reduction='none') * peso_celda).sum()

    perdida = (
        PESO_CLASE * perdida_clase
        + PESO_CONFIANZA * perdida_confianza
        + PESO_CENTRALIDAD * perdida_centralidad
        + PESO_L1 * perdida_l1
        + PESO_CIOU * perdida_ciou
    )

    return perdida


def entrenar_epoca(modelo, cargador, optimizador, criterio_clase, escalador, dispositivo):
    """Recorro una epoca completa de entrenamiento y devuelvo la perdida promedio por imagen."""
    modelo.train()
    perdida_acumulada = 0.0
    muestras = 0

    for imagenes, vector_clases, mapa_confianza, mapa_cajas, _, _ in cargador:
        imagenes = imagenes.to(dispositivo, non_blocking=True)
        vector_clases = vector_clases.to(dispositivo, non_blocking=True)
        mapa_confianza = mapa_confianza.to(dispositivo, non_blocking=True)
        mapa_cajas = mapa_cajas.to(dispositivo, non_blocking=True)

        optimizador.zero_grad(set_to_none=True)

        with torch.autocast(device_type=dispositivo.type, enabled=dispositivo.type == 'cuda'):
            salidas = modelo(imagenes)
        perdida = calcular_perdida(salidas, vector_clases, mapa_confianza, mapa_cajas, criterio_clase)

        escalador.scale(perdida).backward()
        escalador.step(optimizador)
        escalador.update()

        perdida_acumulada += perdida.item() * imagenes.size(0)
        muestras += imagenes.size(0)

    return perdida_acumulada / max(muestras, 1)


@torch.no_grad()
def validar_epoca(modelo, cargador, criterio_clase, dispositivo):
    """Evaluo la validacion y devuelvo la perdida, el F1 macro de la clasificacion de imagen y las metricas de deteccion por objeto."""
    modelo.eval()
    perdida_acumulada = 0.0
    muestras = 0

    verdaderos_positivos = torch.zeros(NUM_CLASES, device=dispositivo)
    falsos_positivos = torch.zeros(NUM_CLASES, device=dispositivo)
    falsos_negativos = torch.zeros(NUM_CLASES, device=dispositivo)
    predicciones = []
    reales = []

    for imagenes, vector_clases, mapa_confianza, mapa_cajas, cajas, clases in cargador:
        imagenes = imagenes.to(dispositivo, non_blocking=True)
        vector_clases = vector_clases.to(dispositivo, non_blocking=True)
        mapa_confianza = mapa_confianza.to(dispositivo, non_blocking=True)
        mapa_cajas = mapa_cajas.to(dispositivo, non_blocking=True)

        with torch.autocast(device_type=dispositivo.type, enabled=dispositivo.type == 'cuda'):
            salidas = modelo(imagenes)
        perdida = calcular_perdida(salidas, vector_clases, mapa_confianza, mapa_cajas, criterio_clase)

        perdida_acumulada += perdida.item() * imagenes.size(0)
        muestras += imagenes.size(0)

        salida_clase, salida_confianza, salida_centralidad, salida_caja = salidas
        prediccion = (torch.sigmoid(salida_clase.float()) >= UMBRAL_CLASE).float()
        verdaderos_positivos += (prediccion * vector_clases).sum(dim=0)
        falsos_positivos += (prediccion * (1 - vector_clases)).sum(dim=0)
        falsos_negativos += ((1 - prediccion) * vector_clases).sum(dim=0)

        # Para el mAP se conservan detecciones de baja confianza; las metricas por objeto filtran despues con UMBRAL_CONFIANZA
        predicciones += postprocesar(
            salida_confianza, salida_centralidad, salida_caja, IMG_SIZE, UMBRAL_CONFIANZA_MAP, UMBRAL_IOU_NMS
        )
        reales += [{'cajas': caja, 'clases': clase} for caja, clase in zip(cajas, clases)]

    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos).clamp(min=1e-6)
    sensibilidad = verdaderos_positivos / (verdaderos_positivos + falsos_negativos).clamp(min=1e-6)
    f1_macro = (2 * precision * sensibilidad / (precision + sensibilidad).clamp(min=1e-6)).mean().item()

    metricas_deteccion = evaluar_detecciones(predicciones, reales, NUM_CLASES, UMBRAL_CONFIANZA)

    return perdida_acumulada / max(muestras, 1), f1_macro, metricas_deteccion


def crear_optimizador(modelo):
    """Separo los parametros en dos grupos para que los bloques descongelados de MobileNetV2 aprendan con una tasa 10 veces menor que las cabezas."""
    parametros_extractor = [parametro for parametro in modelo.extractor.parameters() if parametro.requires_grad]
    parametros_cabezas = [
        parametro for nombre, parametro in modelo.named_parameters()
        if parametro.requires_grad and not nombre.startswith('extractor.')
    ]

    grupos = [{'params': parametros_cabezas, 'lr': TASA_APRENDIZAJE}]
    if parametros_extractor:
        grupos.append({'params': parametros_extractor, 'lr': TASA_APRENDIZAJE * FACTOR_TASA_EXTRACTOR})

    return torch.optim.Adam(grupos)


def entrenar():
    """Ejecuto el entrenamiento completo, guardo el punto de control con mejor mAP@0.5 y corto cuando la validacion deja de mejorar."""
    dispositivo = obtener_dispositivo()
    nombre_gpu = torch.cuda.get_device_name(0) if dispositivo.type == 'cuda' else 'sin GPU'
    print(f"Dispositivo: {dispositivo} | {nombre_gpu}")

    torch.backends.cudnn.benchmark = True
    os.makedirs(os.path.dirname(RUTA_CHECKPOINT), exist_ok=True)

    cargador_entrenamiento, cargador_validacion, dataset_entrenamiento = crear_cargadores()
    peso_clase = calcular_peso_clase(recolectar_etiquetas(dataset_entrenamiento), NUM_CLASES).to(dispositivo)

    modelo = XyronMobileNetV2(NUM_CLASES, DESCONGELAR_DESDE).to(dispositivo)
    criterio_clase = nn.BCEWithLogitsLoss(pos_weight=peso_clase)
    optimizador = crear_optimizador(modelo)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador, mode='max', factor=0.5, patience=PACIENCIA_PLANIFICADOR
    )
    escalador = torch.amp.GradScaler(device=dispositivo.type, enabled=dispositivo.type == 'cuda')
    paro_temprano = ParoTemprano(PACIENCIA, modo='max')

    for epoca in range(1, EPOCAS + 1):
        perdida_entrenamiento = entrenar_epoca(
            modelo, cargador_entrenamiento, optimizador, criterio_clase, escalador, dispositivo
        )
        perdida_validacion, f1_macro, metricas_deteccion = validar_epoca(
            modelo, cargador_validacion, criterio_clase, dispositivo
        )
        map_50 = metricas_deteccion['map_50']
        planificador.step(map_50)

        print(
            f"Epoca {epoca}/{EPOCAS} | perdida train {perdida_entrenamiento:.4f} | "
            f"perdida val {perdida_validacion:.4f} | mAP@0.5 {map_50:.4f} | "
            f"mAP@0.5:0.95 {metricas_deteccion['map_50_95']:.4f} | "
            f"P {metricas_deteccion['precision_global']:.3f} R {metricas_deteccion['sensibilidad_global']:.3f} | "
            f"F1 objetos {metricas_deteccion['f1_macro']:.4f} | F1 imagen {f1_macro:.4f}"
        )

        # Solo valores simples para que torch.load con weights_only pueda leer el punto de control
        metricas = {
            'perdida_validacion': perdida_validacion,
            'f1_macro': f1_macro,
            'map_50': map_50,
            'map_50_95': metricas_deteccion['map_50_95'],
            'f1_macro_objetos': metricas_deteccion['f1_macro'],
            'precision_objetos': metricas_deteccion['precision_global'],
            'sensibilidad_objetos': metricas_deteccion['sensibilidad_global'],
            'descongelar_desde': DESCONGELAR_DESDE,
        }

        if paro_temprano.actualizar(map_50):
            guardar_modelo(modelo, optimizador, epoca, metricas, RUTA_CHECKPOINT)
            print(f"Modelo guardado en {RUTA_CHECKPOINT}")

        if paro_temprano.detener:
            print(f"Paro temprano en la epoca {epoca}")
            break

    return modelo


# Parametros de configuracion
RUTA_DATOS = 'data_1'
RUTA_CHECKPOINT = os.path.join('checkpoints', 'xyron_mnv2_fpn.pt')
IMG_SIZE = 640
NUM_CLASES = 3
BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCAS = 60
TASA_APRENDIZAJE = 0.001
FACTOR_TASA_EXTRACTOR = 0.1
DESCONGELAR_DESDE = 7
PESO_CLASE = 1.0
PESO_CONFIANZA = 1.0
PESO_CENTRALIDAD = 1.0
PESO_L1 = 5.0
PESO_CIOU = 2.0
ALFA_FOCAL = 0.25
GAMMA_FOCAL = 2.0
UMBRAL_CLASE = 0.5
UMBRAL_CONFIANZA = 0.4
UMBRAL_CONFIANZA_MAP = 0.01
UMBRAL_IOU_NMS = 0.5
PACIENCIA = 10
PACIENCIA_PLANIFICADOR = 4

if __name__ == "__main__":
    entrenar()
