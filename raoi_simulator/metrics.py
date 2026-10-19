"""Funciones objetivo del enjambre RAOI en la tarea de farming.

  f1  Balance de carga — desviación estándar de los objetos recogidos por
      cada robot (minimizar).
  f2  Energía promedio — energía total consumida entre el número de robots
      (minimizar).
  f3  Tasa de éxito — objetos entregados entre objetos disponibles
      (maximizar).
  f4  Costo de señalización — costo de los sensores de los robots
      (receptores, radio r_i) más el del emisor del nest (radio r_s)
      (minimizar).

Se calculan a partir de contadores acumulados por robot y del estado final
de los objetos, sin recorrer el reporte completo de la simulación.

Referencia:
  Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
  based on repulsion, attraction and influence. Autonomous Robots.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import numpy as np

from . import config


def farming_metrics(
    robot_counters: np.ndarray,
    n_objects:      int,
    delivered:      np.ndarray,
    sensor_radius:  float,
    emitter_radius: float,
) -> dict:
    """Calcula las cuatro funciones objetivo de la tarea de farming.

      f1 = sqrt( (1/n) * sum_i (o_i - o_mean)^2 ), con o_i los objetos
           recogidos por el robot i.
      f2 = (1/n) * sum_i e_i, con e_i la energía del robot i:
           config.FARMING_ENERGY_COST_SEARCH por iteración buscando +
           config.FARMING_ENERGY_COST_CARRY por iteración cargando.
      f3 = o_delivered / o_total.
      f4 = n · c_i · (r_i / r_0)^p + c_s · (r_s / r_0)^q, con c_i, c_s, p,
           q y r_0 de config.FARMING_SIGNAL_*.

    Args:
        robot_counters: Array (n, 3) por robot: [iteraciones cargando,
                        iteraciones buscando, objetos recogidos].
        n_objects:      Número total de objetos (o_total).
        delivered:      Array booleano (o_total,): True si el objeto fue entregado.
        sensor_radius:  Radio sensor r_i de cada robot hacia el nest (m).
        emitter_radius: Radio de emisión r_s del nest (m).

    Returns:
        Dict con 'load_balance' (f1), 'mean_energy' (f2), 'success_ratio' (f3)
        y 'signaling_cost' (f4).
    """
    carry_iters  = robot_counters[:, 0]
    search_iters = robot_counters[:, 1]
    collected    = robot_counters[:, 2]

    energy = (
        search_iters * config.FARMING_ENERGY_COST_SEARCH
        + carry_iters * config.FARMING_ENERGY_COST_CARRY
    )

    r_0 = config.FARMING_SIGNAL_REFERENCE_RADIUS
    signaling_cost = (
        len(robot_counters) * config.FARMING_SIGNAL_COST_SENSOR
        * (sensor_radius / r_0) ** config.FARMING_SIGNAL_SENSOR_EXPONENT
        + config.FARMING_SIGNAL_COST_EMITTER
        * (emitter_radius / r_0) ** config.FARMING_SIGNAL_EMITTER_EXPONENT
    )

    return {
        "load_balance":   float(np.std(collected)),
        "mean_energy":    float(np.mean(energy)),
        "success_ratio":  int(np.sum(delivered)) / max(n_objects, 1),
        "signaling_cost": float(signaling_cost),
    }
