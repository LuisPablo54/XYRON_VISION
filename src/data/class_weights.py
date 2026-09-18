# Cálculo de pesos por clase para la perdida multietiqueta

# Librerias necesarias
import numpy as np
import torch


def recolectar_etiquetas(dataset):
    """Recorro solo los archivos de etiquetas del dataset y armo la matriz multietiqueta sin cargar ninguna imagen."""
    matriz = np.zeros((len(dataset), dataset.num_clases), dtype=np.float32)

    for indice, nombre in enumerate(dataset.nombres):
        _, clases = dataset.leer_etiquetas(nombre)
        for clase in clases:
            matriz[indice, clase] = 1.0

    return matriz


def calcular_peso_clase(etiquetas, num_clases):
    """Devuelvo el peso de cada clase como negativos entre positivos, que es lo que espera pos_weight de BCEWithLogitsLoss."""
    etiquetas = np.asarray(etiquetas, dtype=np.float32).reshape(-1, num_clases)
    positivos = etiquetas.sum(axis=0)
    negativos = etiquetas.shape[0] - positivos
    peso = np.divide(negativos, positivos, out=np.ones_like(positivos), where=positivos > 0)

    return torch.tensor(peso, dtype=torch.float32)
