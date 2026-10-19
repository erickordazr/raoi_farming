"""Entorno de simulación: paredes del área, parcelas y oclusión de la vista.

Las paredes y las parcelas cercanas se devuelven como puntos (el punto más
cercano de cada una), en el mismo formato que la posición de un vecino
repulsivo, para que behavior.repulsion_vector() los trate igual.

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
    """Detecta las paredes del área dentro del radio de repulsión.

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
    """Detecta los segmentos de parcela dentro del radio de repulsión.

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


def segment_blocked_by_plots(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    plots: list,
) -> bool:
    """Verifica si alguna parcela intersecta la línea entre dos puntos.

    Las parcelas son barreras físicas: bloquean la línea de visión entre
    dos puntos igual que una pared real bloquearía la vista entre dos
    personas a lados opuestos — sea que esos dos puntos sean un robot y un
    objetivo (objeto/nest), o dos robots entre sí. Dado que las parcelas
    son segmentos horizontales (y constante), la intersección se resuelve
    analíticamente en O(1) por parcela:
      t = (y_plot - a.y) / (b.y - a.y)
      x_cross = a.x + t * (b.x - a.x)
      Bloqueado si t está en (0,1) y x_cross en [x0, x1].

    Simétrica en pos_a/pos_b — no importa cuál de los dos puntos se pase
    primero.

    Args:
        pos_a: Posición [x, y] de un extremo de la línea a verificar (m).
        pos_b: Posición [x, y] del otro extremo (m).
        plots: Lista de dicts {'x0', 'x1', 'y'} con los segmentos de parcela.

    Returns:
        True si alguna parcela bloquea la línea entre pos_a y pos_b.
    """
    ax, ay = float(pos_a[0]), float(pos_a[1])
    bx, by = float(pos_b[0]), float(pos_b[1])
    dy     = by - ay

    if abs(dy) < 1e-9:
        return False   # misma Y, ningun segmento horizontal bloquea

    for plot in plots:
        yp = float(plot["y"])
        if (ay < yp < by) or (by < yp < ay):
            t       = (yp - ay) / dy
            x_cross = ax + t * (bx - ax)
            if float(plot["x0"]) <= x_cross <= float(plot["x1"]):
                return True

    return False
