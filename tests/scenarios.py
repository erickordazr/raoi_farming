"""Escenarios de referencia compartidos por las pruebas de caracterización.

Cada escenario fija todos los argumentos de ``farming.run()``, incluida la
semilla, para que la simulación sea reproducible. ``FRAME_STRIDE`` define
cada cuántas iteraciones se guarda un cuadro del reporte en la referencia.
"""

import io
from contextlib import redirect_stdout

import numpy as np

from raoi_simulator import farming

FRAME_STRIDE: int = 10
"""Se guarda un cuadro de cada FRAME_STRIDE iteraciones en la referencia."""

SCENARIOS: dict[str, dict] = {
    # Escenario estándar: pocas parcelas, enjambre mediano.
    "standard": {
        "n_objects": 60,
        "n_plots": 6,
        "individuals": 16,
        "r_r": 0.3,
        "o_r": 1.0,
        "a_r": 2.0,
        "seed": 42,
        "time_limit": 400,
    },
    # Escenario denso: máximo de parcelas, radios de entrega y sensores propios.
    "dense": {
        "n_objects": 120,
        "n_plots": 23,
        "individuals": 24,
        "r_r": 0.3,
        "o_r": 0.6,
        "a_r": 1.0,
        "r_r_deliver": 0.2,
        "o_r_deliver": 0.5,
        "a_r_deliver": 0.8,
        "nest_ri": 4.0,
        "nest_rs": 2.0,
        "obj_r": 1.5,
        "seed": 1,
        "time_limit": 300,
    },
}

METRIC_KEYS: tuple[str, ...] = ("load_balance", "mean_energy", "success_ratio", "signaling_cost")
"""Funciones objetivo f1–f4, en el orden en que se guardan en la referencia."""


def run_scenario(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ejecuta un escenario sin animación y devuelve sus salidas muestreadas.

    Args:
        name: Clave de ``SCENARIOS``.

    Returns:
        Tupla ``(report, objects_report, metrics)``: el reporte del enjambre y
        las posiciones de los objetos cada ``FRAME_STRIDE`` iteraciones (más el
        último cuadro), y las funciones objetivo f1–f4 como arreglo (4,).
    """
    with redirect_stdout(io.StringIO()):
        report, objects_report, metrics = farming.run(animation=False, **SCENARIOS[name])
    frames = np.unique(np.r_[np.arange(0, len(report), FRAME_STRIDE), len(report) - 1])
    values = np.array([metrics[key] for key in METRIC_KEYS], dtype=float)
    return report[frames], objects_report[frames], values
