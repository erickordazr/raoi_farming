# -*- coding: utf-8 -*-
"""
Métricas de desempeño del enjambre RAOI para la tarea de farming.

farming_metrics() es stateless: se calcula enteramente a partir de
collected_objects y delivered (arrays de tamaño individuals/n_objects),
sin recorrer el reporte completo de la simulación — así el costo de
cómputo de las métricas es independiente del número de iteraciones.

Referencia:
  Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
  based on repulsion, attraction and influence. Autonomous Robots.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import numpy as np


def farming_metrics(
    collected_objects: np.ndarray,
    n_objects:         int,
    individuals:       int,
    delivered:         np.ndarray,
) -> dict:
    """
    Calcula las métricas de desempeño de la tarea de farming: f1, f2 y f3.

      f1 — objects_per_robot (cuántos objetos recolectó cada robot) y
           load_balance_std (desviación estándar del balance de carga;
           0 = trabajo perfectamente balanceado entre robots).
      f2 — energy_per_robot (+1 por iteración en búsqueda, +1.5 por
           iteración en entrega/transporte, ya que cargar el objeto cuesta
           más que buscarlo), total_energy y mean_energy (energía total
           del enjambre entre el número de robots).
      f3 — success_fraction: fracción de objetos entregados sobre el total.

    Args:
        collected_objects: Array (N, 4): [delivery_iters, search_iters,
                           n_collected, total_iters] por robot.
        n_objects:         Número total de objetos en el escenario.
        individuals:       Número de robots en el enjambre.
        delivered:         Array booleano (O,): True si el objeto fue entregado.

    Returns:
        Dict con: 'objects_per_robot', 'load_balance_std' (f1),
        'energy_per_robot', 'total_energy', 'mean_energy' (f2),
        'delivered' (int), 'success_fraction' (f3).
    """
    n_delivered      = int(np.sum(delivered))
    success_fraction = n_delivered / max(n_objects, 1)

    objects_per_robot = collected_objects[:, 2].copy()
    load_balance_std  = float(np.std(collected_objects[:, 2]))

    energy_per_robot = (
        collected_objects[:, 1] * 1.0    # iteraciones en busqueda
        + collected_objects[:, 0] * 1.5  # iteraciones cargando (entrega)
    )
    total_energy = float(np.sum(energy_per_robot))
    mean_energy  = total_energy / max(individuals, 1)

    return {
        "objects_per_robot":  objects_per_robot,
        "load_balance_std":   load_balance_std,
        "energy_per_robot":   energy_per_robot,
        "total_energy":       total_energy,
        "mean_energy":        mean_energy,
        "delivered":          n_delivered,
        "success_fraction":   success_fraction,
    }