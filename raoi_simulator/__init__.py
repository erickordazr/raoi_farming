# -*- coding: utf-8 -*-
"""
raoi_simulator — Simulador de enjambre de robots con modelo RAOI para la
tarea de farming (recolección de objetos en parcelas agrícolas).

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

from .farming import (
    run             as farming_run,
    single_run      as farming_single_run,
    statistical_run as farming_statistical_run,
)
from . import config, metrics, behavior, dynamics, environment, visualization

__all__ = [
    "farming_run",
    "farming_single_run",
    "farming_statistical_run",
    "config",
    "metrics",
    "behavior",
    "dynamics",
    "environment",
    "visualization",
]