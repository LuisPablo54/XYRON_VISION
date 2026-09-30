# Lectura del dataset de soldaduras en formato YOLO

# Librerias necesarias
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.xyron_model import RANGOS_ESCALA, generar_puntos


class DatasetSoldadura(Dataset):
    # Leo cada imagen con sus etiquetas YOLO y entrego el vector multietiqueta, todas las cajas y los objetivos por celda de los tres niveles del FPN

    def __init__(self, directorio, num_clases, transformaciones, tamano_imagen=640):
        self.directorio_imagenes = os.path.join(directorio, 'images')
        self.directorio_etiquetas = os.path.join(directorio, 'labels')
        self.num_clases = num_clases
        self.transformaciones = transformaciones
        self.nombres = sorted(os.listdir(self.directorio_imagenes))

        # Centros de todas las celdas (80x80, 40x40 y 20x20 aplanadas) y el rango de distancias que atiende cada nivel
        self.centros, _, self.niveles = generar_puntos(tamano_imagen)
        rangos = torch.tensor(RANGOS_ESCALA) / tamano_imagen
        self.limite_inferior = rangos[self.niveles, 0]
        self.limite_superior = rangos[self.niveles, 1]

    def __len__(self):
        return len(self.nombres)

    def leer_etiquetas(self, nombre_imagen):
        """Devuelvo las cajas normalizadas y sus clases; si el archivo no existe o esta vacio devuelvo listas vacias."""
        ruta = os.path.join(self.directorio_etiquetas, os.path.splitext(nombre_imagen)[0] + '.txt')
        cajas = []
        clases = []

        if not os.path.exists(ruta):
            return cajas, clases

        with open(ruta, 'r') as archivo:
            for linea in archivo:
                valores = linea.split()
                if len(valores) != 5:
                    continue
                clases.append(int(valores[0]))
                cajas.append([float(valor) for valor in valores[1:]])

        return cajas, clases

    def construir_objetivos(self, cajas_esquinas, clases):
        """Asigno como positivas todas las celdas cuyo centro cae dentro de la caja, en el nivel cuyo rango contiene su distancia maxima al borde (asignacion FCOS); entrego los mapas de confianza y de caja por clase."""
        total_puntos = len(self.centros)
        mapa_confianza = torch.zeros(self.num_clases, total_puntos)
        mapa_cajas = torch.zeros(self.num_clases, 4, total_puntos)

        if len(cajas_esquinas) == 0:
            return mapa_confianza, mapa_cajas

        # Distancias izquierda, arriba, derecha y abajo de cada celda a cada caja: [cajas, puntos, 4]
        distancias = torch.stack([
            self.centros[None, :, 0] - cajas_esquinas[:, None, 0],
            self.centros[None, :, 1] - cajas_esquinas[:, None, 1],
            cajas_esquinas[:, None, 2] - self.centros[None, :, 0],
            cajas_esquinas[:, None, 3] - self.centros[None, :, 1],
        ], dim=2)
        dentro = distancias.min(dim=2).values > 0
        distancia_maxima = distancias.max(dim=2).values
        en_rango = (distancia_maxima > self.limite_inferior) & (distancia_maxima <= self.limite_superior)
        positivos = dentro & en_rango

        # Una soldadura larga y delgada puede no caer en ningun rango; en ese caso uso el nivel mas fino donde tenga celdas dentro
        for indice in torch.nonzero(positivos.sum(dim=1) == 0).flatten().tolist():
            if dentro[indice].any():
                nivel_fino = self.niveles[dentro[indice]].min()
                positivos[indice] = dentro[indice] & (self.niveles == nivel_fino)
            else:
                centro_caja = (cajas_esquinas[indice, :2] + cajas_esquinas[indice, 2:]) / 2
                positivos[indice, torch.norm(self.centros - centro_caja, dim=1).argmin()] = True

        # Si dos cajas de la misma clase compiten por una celda, se queda la mas pequena (regla de FCOS)
        areas = (cajas_esquinas[:, 2] - cajas_esquinas[:, 0]) * (cajas_esquinas[:, 3] - cajas_esquinas[:, 1])
        for clase in clases.unique().tolist():
            de_la_clase = clases == clase
            costo = torch.where(positivos[de_la_clase], areas[de_la_clase, None], torch.tensor(float('inf')))
            menor_costo, asignada = costo.min(dim=0)
            celdas_positivas = torch.isfinite(menor_costo)

            mapa_confianza[clase] = celdas_positivas.float()
            mapa_cajas[clase] = cajas_esquinas[de_la_clase][asignada].T * celdas_positivas

        return mapa_confianza, mapa_cajas

    def __getitem__(self, indice):
        nombre = self.nombres[indice]
        imagen = cv2.imread(os.path.join(self.directorio_imagenes, nombre))
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        cajas, clases = self.leer_etiquetas(nombre)

        aumentado = self.transformaciones(image=imagen, bboxes=cajas, clases=clases)
        imagen = aumentado['image']
        cajas = torch.from_numpy(np.asarray(aumentado['bboxes'], dtype=np.float32).reshape(-1, 4))
        clases = torch.from_numpy(np.asarray(aumentado['clases'], dtype=np.int64).reshape(-1))

        vector_clases = torch.zeros(self.num_clases)
        vector_clases[clases] = 1.0

        # Las cajas salen en esquinas normalizadas (x1, y1, x2, y2), que es lo que usan NMS, CIoU y las metricas
        cajas_esquinas = torch.cat([cajas[:, :2] - cajas[:, 2:] / 2, cajas[:, :2] + cajas[:, 2:] / 2], dim=1).clamp(0, 1)
        mapa_confianza, mapa_cajas = self.construir_objetivos(cajas_esquinas, clases)

        return imagen, vector_clases, mapa_confianza, mapa_cajas, cajas_esquinas, clases


def agrupar_lote(muestras):
    """Apilo los tensores de tamano fijo y dejo como listas las cajas y clases, porque cada imagen tiene una cantidad distinta."""
    imagenes, vectores_clases, mapas_confianza, mapas_cajas, cajas, clases = zip(*muestras)

    return (
        torch.stack(imagenes),
        torch.stack(vectores_clases),
        torch.stack(mapas_confianza),
        torch.stack(mapas_cajas),
        list(cajas),
        list(clases),
    )
