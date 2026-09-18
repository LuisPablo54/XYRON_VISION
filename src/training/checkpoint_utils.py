# Guadamos la mejor vesión del model y establecemos criterio de paro

import torch


def guardar_modelo(modelo, optimizador, epoca, metricas, ruta):
    torch.save({
        'epoch': epoca,
        'model_state_dict': modelo.state_dict(),
        'optimizer_state_dict': optimizador.state_dict(),
        'metrics': metricas,
    }, ruta)


def cargar_modelo(modelo, optimizador, ruta, dispositivo):
    """Restauro los pesos guardados y devuelvo la epoca y las metricas del punto de control."""
    punto_control = torch.load(ruta, map_location=dispositivo)
    modelo.load_state_dict(punto_control['model_state_dict'])

    if optimizador is not None:
        optimizador.load_state_dict(punto_control['optimizer_state_dict'])

    return punto_control['epoch'], punto_control['metrics']


class ParoTemprano:
    """Detengo el entrenamiento cuando la perdida de validacion deja de mejorar durante varias epocas seguidas."""

    def __init__(self, paciencia, minima_mejora=0.0):
        self.paciencia = paciencia
        self.minima_mejora = minima_mejora
        self.mejor_valor = None
        self.espera = 0
        self.detener = False

    def actualizar(self, valor):
        """Devuelvo True solo cuando el valor recibido mejora al mejor registrado, que es la senal para guardar el modelo."""
        if self.mejor_valor is None or valor < self.mejor_valor - self.minima_mejora:
            self.mejor_valor = valor
            self.espera = 0
            return True

        self.espera += 1
        self.detener = self.espera >= self.paciencia

        return False
