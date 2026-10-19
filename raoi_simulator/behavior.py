"""Reglas de comportamiento del modelo RAOI.

Percepción de vecinos por zonas (Repulsión, Orientación, Atracción),
composición de los vectores de comportamiento en una dirección de
movimiento y selección del voltaje de las ruedas. La percepción de la
influencia (objeto, nest) es propia de la tarea y está en farming.py.

Todas las funciones son puras: operan sobre el estado que reciben y no
guardan estado propio.

Referencias:
  Ordaz-Rivas et al. (2018). Collective Tasks for a Flock of Robots
  Using Influence Factor. J. Intelligent & Robotic Systems.

  Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
  based on repulsion, attraction and influence. Autonomous Robots.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math
from enum import IntEnum

import numpy as np

from .environment import segment_blocked_by_plots

# ── Zonas ─────────────────────────────────────────────────────────────────────

class Zone(IntEnum):
    """Zona RAOI dominante de un robot; su valor se guarda en la columna 7 del reporte."""

    FREE        = 0   # ninguna zona activa (exploración libre o regreso con brújula)
    REPULSION   = 1
    ATTRACTION  = 2
    ORIENTATION = 3
    INFLUENCE   = 4


# ── Ángulos ───────────────────────────────────────────────────────────────────

def wrap_angle(angle: float) -> float:
    """Normaliza un ángulo al rango [0, 2π).

    Args:
        angle: Ángulo en radianes.

    Returns:
        Ángulo equivalente en [0, 2π).
    """
    return angle % (2 * math.pi)


def angle_diff(
    target: float | np.ndarray,
    current: float | np.ndarray,
) -> float | np.ndarray:
    """Diferencia angular mínima con signo, target - current, en [-π, π).

    Acepta escalares o arrays de NumPy.

    Args:
        target:  Ángulo destino (rad).
        current: Ángulo actual (rad).

    Returns:
        Error angular en [-π, π). Positivo = giro antihorario.
    """
    return (target - current + math.pi) % (2 * math.pi) - math.pi


# ── Vectores de comportamiento ────────────────────────────────────────────────

def repulsion_vector(
    robot_pos: np.ndarray,
    sources: list,
) -> tuple[float, float]:
    """Vector de repulsión del robot respecto a sus fuentes repulsivas.

    Es la suma de los vectores unitarios que apuntan desde cada fuente hacia
    el robot.

    Args:
        robot_pos: Posición [x, y] del robot.
        sources:   Posiciones [x, y] de las fuentes repulsivas (vecinos,
                   puntos de pared y puntos de parcela).

    Returns:
        (vx, vy) sin normalizar; (0, 0) si las fuentes se cancelan.
    """
    vx, vy = 0.0, 0.0
    for pos in sources:
        dx   = robot_pos[0] - pos[0]
        dy   = robot_pos[1] - pos[1]
        norm = max(math.sqrt(dx**2 + dy**2), 1e-9)
        vx  += dx / norm
        vy  += dy / norm
    return vx, vy


def combined_direction(
    current_angle: float,
    active_vectors: dict,
    weights: dict,
) -> float:
    """Dirección de movimiento resultante de las zonas activas.

    Los pesos de las zonas activas se renormalizan para sumar 1. El rumbo
    actual participa como inercia con peso 1 - (peso normalizado máximo):
    si solo hay una zona activa la inercia es cero.

    Args:
        current_angle:  Rumbo actual del robot (rad).
        active_vectors: {zona: (vx, vy)} con vectores unitarios; zonas
                        'R', 'O', 'A', 'I'.
        weights:        Pesos base {'w_r', 'w_o', 'w_a', 'w_I'}.

    Returns:
        Dirección resultante en [0, 2π).
    """
    if not active_vectors:
        return current_angle

    key_map = {"R": "w_r", "O": "w_o", "A": "w_a", "I": "w_I"}
    raw    = {z: weights[key_map[z]] for z in active_vectors}
    total  = max(sum(raw.values()), 1e-9)
    norm_w = {z: w / total for z, w in raw.items()}

    inertia = 1.0 - max(norm_w.values())

    result_x = inertia * math.cos(current_angle)
    result_y = inertia * math.sin(current_angle)
    for zone, (vx, vy) in active_vectors.items():
        result_x += norm_w[zone] * vx
        result_y += norm_w[zone] * vy

    if math.sqrt(result_x**2 + result_y**2) < 1e-9:
        return current_angle

    return wrap_angle(math.atan2(result_y, result_x))


# ── Percepción de vecinos ─────────────────────────────────────────────────────

def detect_neighbors(
    i: int,
    C: np.ndarray,
    repulsion_radius: float,
    orientation_radius: float,
    attraction_radius: float,
    fov: dict,
    wall_count: int,
    plots: list,
) -> dict:
    """Clasifica a los vecinos del robot i en las zonas RAOI.

    Repulsión con prioridad absoluta: se cuenta primero con todos los
    vecinos, y orientación/atracción solo se evalúan si no hay ningún
    vecino en repulsión ni paredes/parcelas en contacto (wall_count == 0).
    Orientación y atracción pueden estar activas a la vez.

    Un vecino cuya línea de vista esté bloqueada por una parcela no se
    percibe en ninguna zona.

    Args:
        i:                  Índice del robot.
        C:                  Estado de todos los robots, shape (N, 6).
        repulsion_radius:   Radio de la zona de repulsión (m).
        orientation_radius: Radio exterior de la zona de orientación (m).
        attraction_radius:  Radio exterior de la zona de atracción (m).
        fov:                Campos de visión por zona (ver config.RAOI_FOV).
        wall_count:         Paredes y parcelas en contacto con el robot.
        plots:              Parcelas {'x0', 'x1', 'y'} que ocluyen la vista.

    Returns:
        Dict con:
          'rep_neighbors'   posiciones [x, y] de los vecinos en repulsión,
          'ox', 'oy'        componentes del rumbo de cada vecino en orientación,
          'ax', 'ay'        componentes unitarias hacia cada vecino en atracción,
          'n_rep', 'n_ori', 'n_att'  número de vecinos en cada zona.
    """
    n = C.shape[0]
    rep_neighbors  = []
    ox, oy, ax, ay = [], [], [], []
    n_rep = n_ori = n_att = 0

    # Geometría de cada vecino visible dentro del radio mayor: (j, distancia,
    # ángulo hacia j, diferencia absoluta entre ese ángulo y el rumbo del
    # robot i). La oclusión solo se calcula para vecinos dentro del radio,
    # porque los demás no caen en ninguna zona.
    max_radius = max(repulsion_radius, orientation_radius, attraction_radius)
    geom = []
    for j in range(n):
        if i == j:
            continue

        dx      = C[j, 0] - C[i, 0]
        dy      = C[j, 1] - C[i, 1]
        dist_ij = math.sqrt(dx**2 + dy**2)
        if dist_ij < 1e-9 or dist_ij > max_radius:
            continue
        if plots and segment_blocked_by_plots(C[i, :2], C[j, :2], plots):
            continue

        angle_to_j = wrap_angle(math.atan2(dy, dx))
        beta       = wrap_angle(angle_to_j - C[i, 3])
        gamma      = wrap_angle(C[i, 3] - angle_to_j)
        geom.append((j, dist_ij, angle_to_j, min(beta, gamma)))

    # Repulsión.
    for j, dist_ij, _, ang_diff in geom:
        if ang_diff < fov["fov_repulsion"] / 2 and dist_ij <= repulsion_radius:
            rep_neighbors.append([C[j, 0], C[j, 1]])
            n_rep += 1

    # Orientación y atracción, solo sin repulsión activa.
    if n_rep == 0 and wall_count == 0:
        for j, dist_ij, angle_to_j, ang_diff in geom:
            if (ang_diff < fov["fov_attraction"] / 2
                    and orientation_radius < dist_ij <= attraction_radius):
                ax.append(math.cos(angle_to_j))
                ay.append(math.sin(angle_to_j))
                n_att += 1

            if (ang_diff < fov["fov_orientation"] / 2
                    and repulsion_radius < dist_ij <= orientation_radius):
                ox.append(math.cos(C[j, 3]))
                oy.append(math.sin(C[j, 3]))
                n_ori += 1

    return {
        "rep_neighbors": rep_neighbors,
        "ox": ox, "oy": oy,
        "ax": ax, "ay": ay,
        "n_rep": n_rep, "n_ori": n_ori, "n_att": n_att,
    }


# ── Voltaje de las ruedas ─────────────────────────────────────────────────────

def select_voltage(
    active_vectors: dict,
    influence_distance: float,
    influence_radius: float,
    noise: float,
    voltages: dict,
) -> np.ndarray:
    """Voltaje de las ruedas según la zona RAOI dominante.

    Repulsión, atracción y orientación usan su voltaje de referencia
    (orientación también cuando no hay ninguna zona activa). Con la
    influencia activa el voltaje se interpola linealmente con la distancia
    al estímulo: voltaje de repulsión encima del estímulo y de atracción en
    el borde de detección, de modo que el robot desacelera al acercarse.

    Ambas ruedas reciben el mismo voltaje: el voltaje fija la velocidad de
    avance y el giro se resuelve de forma cinemática (ver farming.run()).

    Args:
        active_vectors:     Zonas RAOI activas del robot.
        influence_distance: Distancia percibida al estímulo (m).
        influence_radius:   Radio de detección del estímulo (m).
        noise:              Ruido de actuador de esta iteración (V).
        voltages:           Voltajes de referencia {'repulsion',
                            'orientation', 'attraction'} (V).

    Returns:
        Voltajes [u_r, u_l], shape (2,).
    """
    v_rep = voltages["repulsion"]
    v_att = voltages["attraction"]

    if "R" in active_vectors:
        v_base = v_rep + noise
    elif "I" in active_vectors:
        # Fracción de la distancia al estímulo respecto al radio de detección.
        frac   = influence_distance / (max(influence_radius, 1e-9) + 1e-9)
        v_base = frac * (v_att - v_rep) + v_rep + noise
    elif "A" in active_vectors:
        v_base = v_att + noise
    else:
        v_base = voltages["orientation"] + noise

    u = float(np.clip(v_base, v_rep, v_att))
    return np.array([u, u])
