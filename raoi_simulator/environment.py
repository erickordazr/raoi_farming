# -*- coding: utf-8 -*-
"""
Entorno de simulación: detección de paredes y gestión del escenario.

Las paredes se tratan como fuentes de repulsión virtual — sus puntos más
cercanos se añaden a la misma lista que los vecinos repulsivos del robot,
lo que garantiza consistencia con el modelo RAOI sin lógica especial.

Para añadir elementos al entorno (obstáculos estáticos, zonas objetivo,
fuentes de influencia múltiples): agregar funciones aquí. La simulación
principal solo necesita llamar estas funciones y pasar los resultados
al módulo behavior.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math
import numpy as np


def detect_walls(
    pos: np.ndarray,
    repulsion_radius: float,
    area_limits: float,
) -> list:
    """
    Detecta las paredes del área dentro del radio de repulsión.

    Devuelve los puntos más cercanos de cada pared detectada como posiciones
    virtuales, idénticas en formato a las posiciones de robots vecinos. Esto
    permite que behavior.repulsion_vector() trate paredes y robots de la misma
    forma, sin código de caso especial.

    Paredes consideradas: sur (y=0), norte (y=L), oeste (x=0), este (x=L).
    En una esquina pueden detectarse hasta 2 paredes simultáneamente.

    Args:
        pos:              Posición [x, y] del robot (m).
        repulsion_radius: Radio de repulsión efectivo (m).
        area_limits:      Lado del área cuadrada (m).

    Returns:
        Lista de posiciones virtuales [[x, y], ...] de paredes detectadas.
        Vacía si ninguna pared está dentro del radio.
    """
    wall_points = []
    x, y = float(pos[0]), float(pos[1])

    if y < repulsion_radius:                    # pared sur
        wall_points.append([x, 0.0])
    if y > area_limits - repulsion_radius:      # pared norte
        wall_points.append([x, area_limits])
    if x < repulsion_radius:                    # pared oeste
        wall_points.append([0.0, y])
    if x > area_limits - repulsion_radius:      # pared este
        wall_points.append([area_limits, y])

    return wall_points


def detect_plots(
    pos: np.ndarray,
    plots: list,
    repulsion_radius: float,
) -> list:
    """
    Detecta los segmentos de parcela dentro del radio de repulsión.

    Cada parcela es un segmento horizontal definido por sus dos extremos
    (x0, y_plot) → (x1, y_plot). Para cada segmento se calcula el punto
    más cercano al robot (proyección ortogonal clampada al segmento) y,
    si está dentro del radio de repulsión, se añade a la lista de fuentes
    repulsivas en el mismo formato que detect_walls().

    Esto permite que behavior.repulsion_vector() trate parcelas y paredes
    de forma idéntica, sin ningún código de caso especial.

    Args:
        pos:              Posición [x, y] del robot (m).
        plots:            Lista de dicts {'x0', 'x1', 'y'} con los segmentos.
        repulsion_radius: Radio de repulsión efectivo (m).

    Returns:
        Lista de posiciones virtuales [[x, y], ...] de segmentos detectados.
        Vacía si ninguna parcela está dentro del radio.
    """
    plot_points = []
    rx, ry = float(pos[0]), float(pos[1])

    for plot in plots:
        x0 = float(plot["x0"])
        x1 = float(plot["x1"])
        yp = float(plot["y"])

        # Punto más cercano sobre el segmento (proyección clampada)
        t   = max(0.0, min(1.0, (rx - x0) / max(x1 - x0, 1e-9)))
        cx  = x0 + t * (x1 - x0)
        cy  = yp

        dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
        if dist <= repulsion_radius:
            plot_points.append([cx, cy])

    return plot_points