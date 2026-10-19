"""Simulador de enjambre de robots con el modelo RAOI para la tarea de farming.

El enjambre recolecta objetos distribuidos en franjas junto a parcelas
agrícolas y los deposita en el nido.

Uso básico:
    from raoi_simulator.farming import single_run, statistical_run
    from raoi_simulator import config

Autores:
    Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
    FIME — Universidad Autónoma de Nuevo León

Referencia principal:
    Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
    based on repulsion, attraction and influence. Autonomous Robots.
"""

import importlib
from types import ModuleType

from . import behavior, config, dynamics, environment, metrics
from .farming import (
    run as farming_run,
    single_run as farming_single_run,
    statistical_run as farming_statistical_run,
)

__all__ = [
    "behavior",
    "config",
    "dynamics",
    "environment",
    "farming_run",
    "farming_single_run",
    "farming_statistical_run",
    "metrics",
    "visualization",
]


def __getattr__(name: str) -> ModuleType:
    """Importa la visualización solo cuando se usa (requiere pygame y OpenCV).

    Args:
        name: Atributo solicitado del paquete.

    Returns:
        El módulo raoi_simulator.visualization si name == "visualization".

    Raises:
        AttributeError: Si el atributo no existe.
    """
    if name == "visualization":
        return importlib.import_module(f"{__name__}.visualization")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
