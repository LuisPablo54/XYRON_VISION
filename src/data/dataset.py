# Lectura del dataset de soldaduras en formato YOLO

# Librerias necesarias
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetSoldadura(Dataset):
    """Leo cada imagen con sus etiquetas YOLO y entrego el vector multietiqueta de clases junto con la caja de mayor area."""

    def __init__(self, directorio, num_clases, transformaciones):
        self.directorio_imagenes = os.path.join(directorio, 'images')
        self.directorio_etiquetas = os.path.join(directorio, 'labels')
        self.num_clases = num_clases
        self.transformaciones = transformaciones
        self.nombres = sorted(os.listdir(self.directorio_imagenes))

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

    def __getitem__(self, indice):
        nombre = self.nombres[indice]
        imagen = cv2.imread(os.path.join(self.directorio_imagenes, nombre))
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        cajas, clases = self.leer_etiquetas(nombre)

        aumentado = self.transformaciones(image=imagen, bboxes=cajas, clases=clases)
        imagen = aumentado['image']
        cajas = np.asarray(aumentado['bboxes'], dtype=np.float32).reshape(-1, 4)
        clases = np.asarray(aumentado['clases'], dtype=np.int64).reshape(-1)

        vector_clases = torch.zeros(self.num_clases)
        vector_clases[torch.from_numpy(clases)] = 1.0

        # La rotacion puede dejar la imagen sin cajas visibles, la mascara evita que esos casos contaminen la regresion
        caja_principal = torch.zeros(4)
        mascara_caja = torch.zeros(1)
        if len(cajas) > 0:
            areas = cajas[:, 2] * cajas[:, 3]
            caja_principal = torch.from_numpy(cajas[int(np.argmax(areas))])
            mascara_caja = torch.ones(1)

        return imagen, vector_clases, caja_principal, mascara_caja
