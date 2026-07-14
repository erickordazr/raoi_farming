"""
Reglas de comportamiento del modelo RAOI.

Implementa las reglas de percepción y acción del modelo RAOI (Repulsión,
Atracción, Orientación, Influencia): detección de vecinos, composición de
vectores de movimiento y selección de voltaje de ruedas. La detección de
la fuente de influencia es específica de cada tarea (ver
_detect_influence_farming en farming.py, con oclusión de visión por
parcelas) y no vive en este módulo.

Cada función es stateless: opera sobre el estado actual del enjambre y
devuelve vectores o voltajes, sin guardar ningún estado propio.

Referencia:
  Ordaz-Rivas et al. (2018). Collective Tasks for a Flock of Robots
  Using Influence Factor. J. Intelligent & Robotic Systems.

  Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
  based on repulsion, attraction and influence. Autonomous Robots.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math

import numpy as np

from . import config


# ── Utilidades de ángulos ─────────────────────────────────────────────────────

def wrap_angle(angle: float) -> float:
    """
    Normaliza un ángulo al rango [0, 2π].

    Args:
        angle: Ángulo en radianes (cualquier valor).

    Returns:
        Ángulo equivalente en [0, 2π].
    """
    return angle % (2 * math.pi)


def angle_diff(target: float, current: float) -> float:
    """
    Diferencia angular mínima con signo entre target y current.

    Args:
        target:  Ángulo deseado (rad).
        current: Ángulo actual (rad).

    Returns:
        Error angular en [-π, π]. Positivo → giro antihorario.
    """
    return (target - current + math.pi) % (2 * math.pi) - math.pi


# ── Utilidades de normalización ───────────────────────────────────────────────

def normalize(value: float, lb: float, ub: float) -> float:
    """
    Normaliza un valor al rango [0, 1].

    Args:
        value: Valor a normalizar.
        lb:    Límite inferior.
        ub:    Límite superior.

    Returns:
        Valor normalizado, con guarda contra división por cero.
    """
    return (value - lb) / ((ub - lb) + 1e-9)


def denormalize(n: float, lb: float, ub: float) -> float:
    """
    Mapea un valor normalizado de [0, 1] al rango [lb, ub].

    Args:
        n:  Valor normalizado.
        lb: Límite inferior.
        ub: Límite superior.

    Returns:
        Valor en [lb, ub].
    """
    return n * (ub - lb) + lb


# ── Vectores de comportamiento RAOI ──────────────────────────────────────────

def repulsion_vector(
    robot_pos: np.ndarray,
    neighbor_positions: list,
) -> tuple[float, float]:
    """
    Vector de repulsión como suma de (p_i - p_j) normalizados individualmente.

    Cada fuente repulsiva (vecino o pared virtual) aporta un vector unitario
    que apunta desde esa fuente hacia el robot. La suma ponderada da la
    dirección de escape resultante.

    Args:
        robot_pos:          Posición del robot i, shape (2,).
        neighbor_positions: Lista de posiciones [x, y] de fuentes repulsivas.
                            Incluye tanto robots vecinos como puntos de pared.

    Returns:
        (vx, vy): Vector de repulsión resultante. Puede ser (0, 0)
                  si las fuentes se cancelan exactamente.
    """
    vx, vy = 0.0, 0.0
    for pos in neighbor_positions:
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
    """
    Calcula la dirección de movimiento resultante combinando las zonas activas.

    Los pesos de las zonas activas se renormalizan dinámicamente para sumar 1,
    preservando sus proporciones relativas. La inercia del robot participa
    con peso (1 - max_weight): con w_r=1.0 la inercia es cero (prioridad
    absoluta); con pesos parciales, amortigua cambios bruscos.

    Args:
        current_angle:  Orientación actual del robot (rad).
        active_vectors: Dict zona → (vx, vy) de vectores unitarios activos.
                        Claves válidas: 'R' (repulsión), 'O' (orientación),
                        'A' (atracción), 'I' (influencia).
        weights:        Dict de pesos base por zona (claves 'w_r', 'w_o',
                        'w_a', 'w_I'), por ejemplo config.FARMING_RAOI_WEIGHTS.

    Returns:
        Ángulo resultante en [0, 2π].
    """
    if not active_vectors:
        return current_angle

    key_map = {"R": "w_r", "O": "w_o", "A": "w_a", "I": "w_I"}
    raw    = {z: weights[key_map[z]] for z in active_vectors}
    total  = max(sum(raw.values()), 1e-9)
    norm_w = {z: w / total for z, w in raw.items()}

    max_w   = max(norm_w.values())
    inertia = 1.0 - max_w

    result_x = inertia * math.cos(current_angle)
    result_y = inertia * math.sin(current_angle)
    for zone, (vx, vy) in active_vectors.items():
        result_x += norm_w[zone] * vx
        result_y += norm_w[zone] * vy

    if math.sqrt(result_x**2 + result_y**2) < 1e-9:
        return current_angle

    return wrap_angle(math.atan2(result_y, result_x))


# ── Sensado de vecinos ────────────────────────────────────────────────────────

def detect_neighbors(
    i: int,
    C: np.ndarray,
    repulsion_radius: float,
    orientation_radius: float,
    attraction_radius: float,
    fov: dict,
    wall_count: int,
) -> dict:
    """
    Detecta los vecinos del robot i en cada zona de percepción RAOI.

    La zona de atracción solo se activa si no hay vecinos repulsivos ni paredes
    detectadas. La zona de orientación tiene prioridad sobre la atracción y
    ambas pueden coexistir sin suprimirse mutuamente.

    Args:
        i:                  Índice del robot en el enjambre.
        C:                  Estado de todos los robots, shape (N, 6).
        repulsion_radius:   Radio efectivo de repulsión (m).
        orientation_radius: Radio efectivo de orientación (m).
        attraction_radius:  Radio efectivo de atracción (m).
        fov:                Dict de ángulos de campo de visión por zona.
        wall_count:         Número de paredes detectadas (suprime O y A si > 0).

    Returns:
        Dict con:
          'rep_neighbors' : lista de posiciones [x, y] en zona R.
          'ox', 'oy'      : componentes del vector de orientación (listas).
          'ax', 'ay'      : componentes del vector de atracción (listas).
          'n_rep'         : número de vecinos en zona R.
          'n_ori'         : número de vecinos en zona O.
          'n_att'         : número de vecinos en zona A.
    """
    n     = C.shape[0]
    rep_neighbors          = []
    ox, oy, ax, ay         = [], [], [], []
    n_rep = n_ori = n_att  = 0

    for j in range(n):
        if i == j:
            continue

        dx      = C[j, 0] - C[i, 0]
        dy      = C[j, 1] - C[i, 1]
        dist_ij = math.sqrt(dx**2 + dy**2)
        if dist_ij < 1e-9:
            continue

        angle_to_j = wrap_angle(math.atan2(dy, dx))
        beta       = wrap_angle(angle_to_j - C[i, 3])
        gamma      = wrap_angle(C[i, 3] - angle_to_j)
        ang_diff   = min(beta, gamma)

        d_rep = dist_ij if ang_diff < fov["fov_repulsion"]   / 2 else math.inf
        d_att = dist_ij if ang_diff < fov["fov_attraction"]  / 2 else math.inf
        d_ori = dist_ij if ang_diff < fov["fov_orientation"] / 2 else math.inf

        if d_rep <= repulsion_radius:
            rep_neighbors.append([C[j, 0], C[j, 1]])
            n_rep += 1

        if (orientation_radius < d_att <= attraction_radius
                and n_rep == 0 and wall_count == 0):
            ax.append(math.cos(angle_to_j))
            ay.append(math.sin(angle_to_j))
            n_att += 1

        if (repulsion_radius < d_ori <= orientation_radius
                and n_rep == 0 and wall_count == 0):
            ox.append(math.cos(C[j, 3]))
            oy.append(math.sin(C[j, 3]))
            n_ori += 1

    return {
        "rep_neighbors": rep_neighbors,
        "ox": ox, "oy": oy,
        "ax": ax, "ay": ay,
        "n_rep": n_rep, "n_ori": n_ori, "n_att": n_att,
    }
# ── Selección de voltaje ──────────────────────────────────────────────────────

def select_voltage(
    active_vectors: dict,
    desired_theta: float,
    theta_before_raoi: float,
    influence_distance: float,
    influence_radius: float,
    noise: float,
    voltages: dict,
    stimulus_radius: float = 0.0,
) -> np.ndarray:
    """
    Calcula los voltajes de rueda [u_r, u_l] para el robot.

    El voltaje base se selecciona según la zona RAOI dominante. Cuando la
    zona I está activa, el voltaje se escala con la distancia normalizada
    al estímulo usando el rango efectivo r_I + r_s (influence_radius +
    stimulus_radius) como referencia: voltaje mínimo en el borde de
    detección (dist = r_I + r_s) y voltaje máximo encima del estímulo
    (dist = 0). Con stimulus_radius = 0 se recupera el comportamiento original.

    Args:
        active_vectors:    Zonas RAOI activas del robot.
        desired_theta:     Dirección de movimiento deseada (rad).
        theta_before_raoi: Orientación antes de aplicar RAOI (rad).
        influence_distance: Distancia real a la fuente de influencia (m).
        influence_radius:   Radio sensorial del robot r_I (m).
        noise:             Ruido blanco del paso actual (V).
        voltages:          Dict de voltajes de referencia (config.VOLTAGE).
        stimulus_radius:   Radio físico del estímulo r_s (m). Default 0.

    Returns:
        Array [u_r, u_l], shape (2,).
    """
    if "R" in active_vectors:
        v_base = voltages["repulsion"] + noise
    elif "I" in active_vectors:
        effective_range = influence_radius + stimulus_radius
        n      = normalize(influence_distance, 0.0, max(effective_range, 1e-9))
        v_base = denormalize(n, voltages["repulsion"], voltages["attraction"]) + noise
    elif "A" in active_vectors:
        v_base = voltages["attraction"] + noise
    else:
        v_base = voltages["orientation"] + noise

    theta_err = angle_diff(desired_theta, theta_before_raoi)
    v_diff    = config.KP_TURN * theta_err

    v_min = voltages["repulsion"]
    v_max = voltages["attraction"]
    u_r   = float(np.clip(v_base + v_diff, v_min, v_max))
    u_l   = float(np.clip(v_base - v_diff, v_min, v_max))

    return np.array([u_r, u_l])