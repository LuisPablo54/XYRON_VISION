# Metricas por objeto y por imagen para la deteccion de soldaduras

# Librerias necesarias
import numpy as np
import torch
from torchvision.ops import box_iou


def emparejar_detecciones(cajas_predichas, puntajes, clases_predichas, cajas_reales, clases_reales, umbral_iou):
    """Emparejo por clase empezando por la prediccion con mas confianza; cada caja real se usa una sola vez, asi un duplicado de la misma clase queda como FP."""
    es_verdadero_positivo = torch.zeros(len(cajas_predichas), dtype=torch.bool)
    real_detectada = torch.zeros(len(cajas_reales), dtype=torch.bool)

    if len(cajas_predichas) == 0 or len(cajas_reales) == 0:
        return es_verdadero_positivo, real_detectada

    # Un Bad Weld y un Defect en la misma zona no compiten entre si porque solo se comparan cajas de la misma clase
    matriz_iou = box_iou(cajas_predichas, cajas_reales)
    matriz_iou[clases_predichas[:, None] != clases_reales[None, :]] = -1.0

    for indice in torch.argsort(puntajes, descending=True, stable=True).tolist():
        iou_disponible = matriz_iou[indice].masked_fill(real_detectada, -1.0)
        mejor_iou, mejor_real = iou_disponible.max(dim=0)
        if mejor_iou >= umbral_iou:
            es_verdadero_positivo[indice] = True
            real_detectada[mejor_real] = True

    return es_verdadero_positivo, real_detectada


def precision_promedio(puntajes, verdaderos_positivos, total_reales):
    """Calculo la AP con la interpolacion de 101 puntos de COCO a partir de las predicciones acumuladas de una clase."""
    if total_reales == 0:
        return float('nan')
    if len(puntajes) == 0:
        return 0.0

    orden = np.argsort(-puntajes, kind='mergesort')
    aciertos = np.cumsum(verdaderos_positivos[orden])
    errores = np.cumsum(~verdaderos_positivos[orden])
    sensibilidad = aciertos / total_reales
    precision = aciertos / (aciertos + errores)

    # Envolvente: la precision en cada punto es la maxima alcanzable con una sensibilidad igual o mayor
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    posiciones = np.searchsorted(sensibilidad, np.linspace(0, 1, 101), side='left')
    precision_interpolada = np.where(posiciones < len(precision), precision[np.minimum(posiciones, len(precision) - 1)], 0.0)

    return float(precision_interpolada.mean())


def evaluar_detecciones(predicciones, reales, num_clases, umbral_confianza):
    """Devuelvo el mAP@0.5, el mAP@0.5:0.95, las metricas por clase al umbral de confianza y las metricas por imagen."""
    umbrales_iou = UMBRALES_IOU_COCO
    puntajes_por_clase = [[] for _ in range(num_clases)]
    aciertos_por_umbral = [[[] for _ in range(num_clases)] for _ in umbrales_iou]
    total_reales = np.zeros(num_clases, dtype=np.int64)

    verdaderos_positivos = np.zeros(num_clases, dtype=np.int64)
    falsos_positivos = np.zeros(num_clases, dtype=np.int64)
    imagenes_completas = 0
    imagenes_con_reales = 0
    errores_conteo = []

    for prediccion, real in zip(predicciones, reales):
        cajas_predichas = prediccion['cajas'].float().cpu()
        puntajes = prediccion['puntajes'].float().cpu()
        clases_predichas = prediccion['clases'].cpu()
        cajas_reales = real['cajas'].float().cpu()
        clases_reales = real['clases'].cpu()

        total_reales += np.bincount(clases_reales.numpy(), minlength=num_clases)
        for clase in range(num_clases):
            puntajes_por_clase[clase].append(puntajes[clases_predichas == clase].numpy())

        for posicion, umbral_iou in enumerate(umbrales_iou):
            es_verdadero_positivo, _ = emparejar_detecciones(
                cajas_predichas, puntajes, clases_predichas, cajas_reales, clases_reales, umbral_iou
            )
            for clase in range(num_clases):
                aciertos_por_umbral[posicion][clase].append(es_verdadero_positivo[clases_predichas == clase].numpy())

        # Metricas al umbral de confianza de operacion y con IoU >= 0.5
        seleccion = puntajes >= umbral_confianza
        es_verdadero_positivo, real_detectada = emparejar_detecciones(
            cajas_predichas[seleccion], puntajes[seleccion], clases_predichas[seleccion],
            cajas_reales, clases_reales, UMBRAL_IOU_EMPAREJAMIENTO,
        )
        clases_seleccionadas = clases_predichas[seleccion].numpy()
        verdaderos_positivos += np.bincount(clases_seleccionadas[es_verdadero_positivo.numpy()], minlength=num_clases)
        falsos_positivos += np.bincount(clases_seleccionadas[~es_verdadero_positivo.numpy()], minlength=num_clases)

        if len(cajas_reales) > 0:
            imagenes_con_reales += 1
            imagenes_completas += int(real_detectada.all())
        errores_conteo.append(abs(int(seleccion.sum()) - len(cajas_reales)))

    ap_por_umbral = np.array([
        [
            precision_promedio(
                np.concatenate(puntajes_por_clase[clase]),
                np.concatenate(aciertos_por_umbral[posicion][clase]).astype(bool),
                total_reales[clase],
            )
            for clase in range(num_clases)
        ]
        for posicion in range(len(umbrales_iou))
    ])

    falsos_negativos = total_reales - verdaderos_positivos
    precision = verdaderos_positivos / np.maximum(verdaderos_positivos + falsos_positivos, 1)
    sensibilidad = verdaderos_positivos / np.maximum(total_reales, 1)
    f1 = 2 * precision * sensibilidad / np.maximum(precision + sensibilidad, 1e-6)
    total_verdaderos = verdaderos_positivos.sum()

    return {
        'map_50': float(np.nanmean(ap_por_umbral[0])),
        'map_50_95': float(np.nanmean(ap_por_umbral)),
        'ap_50_por_clase': ap_por_umbral[0],
        'verdaderos_positivos': verdaderos_positivos,
        'falsos_positivos': falsos_positivos,
        'falsos_negativos': falsos_negativos,
        'precision_por_clase': precision,
        'sensibilidad_por_clase': sensibilidad,
        'f1_por_clase': f1,
        'f1_macro': float(f1.mean()),
        'precision_global': float(total_verdaderos / max(total_verdaderos + falsos_positivos.sum(), 1)),
        'sensibilidad_global': float(total_verdaderos / max(total_reales.sum(), 1)),
        'imagenes_completas': imagenes_completas / max(imagenes_con_reales, 1),
        'error_conteo_medio': float(np.mean(errores_conteo)) if errores_conteo else 0.0,
    }


# Umbrales de IoU de COCO para el mAP@0.5:0.95; el primero corresponde al mAP@0.5
UMBRALES_IOU_COCO = np.linspace(0.5, 0.95, 10)

# IoU minima para que una prediccion cuente como acierto en las metricas por objeto
UMBRAL_IOU_EMPAREJAMIENTO = 0.5
