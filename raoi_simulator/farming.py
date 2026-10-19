"""Simulación de la tarea de farming (recolección agrícola) con el modelo RAOI.

El enjambre recolecta objetos repartidos en franjas a ambos lados de
parcelas agrícolas (segmentos horizontales) y los deposita en el nest.

Escenario:
  Parcelas de longitud FARMING_PLOT_LENGTH, simétricas respecto al centro
  del área y separadas FARMING_PLOT_SEPARATION, recorridas hacia el borde
  este; el pasillo libre del lado oeste contiene la línea de arranque y el
  nest. El número máximo de parcelas se calcula a partir de la geometría.

Percepción:
  Un robot no percibe un objeto, el nest ni a otro robot (para R/O/A) si
  una parcela se interpone en la línea que los une
  (environment.segment_blocked_by_plots).

Influencia:
  - Sin objeto: hacia el objeto visible más cercano, dentro del alcance y
    del cono frontal del sensor de objetos. Sin objeto visible, exploración
    tipo vuelo de Lévy.
  - Con objeto: hacia el nest, dentro de su radio de detección (r_i + r_s).

Regreso con brújula (menor prioridad que cualquier zona RAOI):
  Un robot cargado que no percibe nada se guía con una brújula que solo
  indica el lado de la arena donde está el nest (oeste), con sesgo fijo y
  ruido por lectura. Si toca la cerca oeste sin percibir el nest, lo busca
  en vaivén creciente a lo largo de ella (tramos L, 2L, 3L, ... alternando
  norte y sur y regresando a la cerca entre tramos), como la búsqueda
  sistemática de las hormigas del desierto.

Movimiento:
  Las zonas RAOI fijan la dirección deseada y el voltaje (velocidad de
  avance). El rumbo se ajusta de forma cinemática hacia la dirección
  deseada (promedio 50/50, con tope de giro, válvula de escape y
  aproximación final cerca de un objetivo puntual): es un supuesto del
  modelo, el giro no pasa por la dinámica de los motores. El contacto con
  paredes y parcelas es deslizante. Recoger y depositar se evalúan sobre
  el tramo recorrido en cada iteración.

La simulación termina al entregar todos los objetos o al agotar max_iter.

Referencia:
  Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
  based on repulsion, attraction and influence. Autonomous Robots.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import config, console
from .behavior import (
    Zone,
    angle_diff,
    combined_direction,
    detect_neighbors,
    repulsion_vector,
    select_voltage,
    wrap_angle,
)
from .dynamics import DynamicsConstants, resolve_walls, step as dynamics_step
from .environment import detect_plots, detect_walls, segment_blocked_by_plots
from .metrics import farming_metrics

logger = logging.getLogger(__name__)

# Fases del regreso con brújula de un robot cargado.
HOME_WEST = 0   # rumbo oeste hasta tocar la cerca oeste
HOME_LEG  = 1   # tramo norte-sur de la búsqueda en vaivén

_EPS = 1e-9                  # tolerancia para normas, distancias y desplazamientos (m)
_EPS_SQ = 1e-12              # tolerancia para longitudes al cuadrado (m²)
_MIN_STRIP_HEIGHT = 0.1      # alto mínimo (m) de la franja superior de objetos
_SPAWN_WALL_CLEARANCE = 0.1  # holgura (m) entre la columna de arranque y la pared oeste


# ============================================================================
# Geometría del escenario
# ============================================================================

def _compute_max_plots(
    area_limits:    float,
    separation:     float,
    strip_width:    float,
    plot_repulsion: float,
    margin_y:       float,
) -> int:
    """Número máximo de parcelas que caben en el área.

    Un bloque de n parcelas ocupa (n-1)·separation + 2·(strip_width +
    plot_repulsion) en Y, y debe caber en area_limits - 2·margin_y.

    Args:
        area_limits:    Lado del área cuadrada (m).
        separation:     Distancia entre ejes de parcelas (m).
        strip_width:    Ancho de la franja de objetos a cada lado (m).
        plot_repulsion: Radio de repulsión de las parcelas (m).
        margin_y:       Margen sin parcelas en cada extremo en Y (m).

    Returns:
        Número máximo de parcelas (al menos 1).
    """
    available = area_limits - 2.0 * margin_y
    n = 1
    while True:
        block = (n - 1) * separation + 2.0 * (strip_width + plot_repulsion)
        if block > available:
            return max(1, n - 1)
        n += 1


def max_plots() -> int:
    """Número máximo de parcelas que caben con la geometría de config.py.

    Returns:
        Número máximo de parcelas (al menos 1).
    """
    return _compute_max_plots(
        config.FARMING_AREA_LIMITS, config.FARMING_PLOT_SEPARATION,
        config.FARMING_STRIP_WIDTH, config.FARMING_PLOT_REPULSION,
        config.FARMING_PLOT_MARGIN_Y,
    )


def _build_plots_symmetric(
    n_plots:     int,
    plot_length: float,
    area_limits: float,
    separation:  float,
    east_margin: float,
) -> list[dict]:
    """Genera las parcelas, simétricas en Y y recorridas hacia el borde este.

    Con n_plots impar hay una parcela central; con n_plots par, pares
    simétricos alrededor del centro del área.

    Args:
        n_plots:     Número de parcelas.
        plot_length: Longitud de cada parcela (m).
        area_limits: Lado del área cuadrada (m).
        separation:  Distancia entre ejes de parcelas (m).
        east_margin: Distancia del extremo este de las parcelas a la pared este (m).

    Returns:
        Lista de dicts {'x0', 'x1', 'y'} ordenada por y creciente.
    """
    center_y = area_limits / 2.0
    x1 = area_limits - east_margin
    x0 = x1 - plot_length

    ys = []
    if n_plots % 2 == 1:
        ys.append(center_y)
        for k in range(1, n_plots // 2 + 1):
            ys.append(center_y + k * separation)
            ys.append(center_y - k * separation)
    else:
        for k in range(1, n_plots // 2 + 1):
            offset = (k - 0.5) * separation
            ys.append(center_y + offset)
            ys.append(center_y - offset)

    ys.sort()
    return [{"x0": x0, "x1": x1, "y": y} for y in ys]


def _spawn_objects_farming(
    n_objects:      int,
    plots:          list[dict],
    strip_width:    float,
    plot_repulsion: float,
    area_limits:    float,
    rng:            np.random.Generator,
) -> np.ndarray:
    """Reparte los objetos uniformemente entre parcelas, en franjas a ambos lados.

    Cada parcela recibe n_objects // n_plots objetos (el residuo va a las
    primeras parcelas), la mitad arriba y la mitad abajo de la parcela. Cada
    franja empieza a plot_repulsion + config.FARMING_STRIP_GAP del eje.

    Args:
        n_objects:      Número total de objetos.
        plots:          Parcelas {'x0', 'x1', 'y'}.
        strip_width:    Ancho de la franja a cada lado (m).
        plot_repulsion: Radio de repulsión de las parcelas (m).
        area_limits:    Lado del área cuadrada (m).
        rng:            Generador aleatorio.

    Returns:
        Posiciones de los objetos, shape (n_objects, 2).
    """
    n_plots = len(plots)
    margin  = plot_repulsion + config.FARMING_STRIP_GAP
    base    = n_objects // n_plots
    extra   = n_objects  % n_plots

    positions = []
    for p_idx, plot in enumerate(plots):
        n_here = base + (1 if p_idx < extra else 0)
        x0, x1 = plot["x0"], plot["x1"]
        yp     = plot["y"]

        y_lo_up = yp + margin
        y_hi_up = min(yp + margin + strip_width, area_limits - margin)
        y_lo_dn = max(yp - margin - strip_width, margin)
        y_hi_dn = yp - margin

        n_up = n_here // 2
        n_dn = n_here - n_up

        for _ in range(n_up):
            positions.append([
                rng.uniform(x0, x1),
                rng.uniform(y_lo_up, max(y_lo_up + _MIN_STRIP_HEIGHT, y_hi_up)),
            ])
        for _ in range(n_dn):
            positions.append([
                rng.uniform(x0, x1),
                rng.uniform(max(y_lo_dn, margin), y_hi_dn),
            ])

    return np.array(positions, dtype=float).reshape(-1, 2)


# ============================================================================
# Contacto y percepción
# ============================================================================

def _resolve_plot_collision(
    c_old:           np.ndarray,
    c_new:           np.ndarray,
    plots:           list[dict],
    preferred_dir_x: float,
) -> tuple[np.ndarray, bool]:
    """Impide que el robot atraviese una parcela (contacto deslizante).

    Si el paso c_old -> c_new cruza el segmento de una parcela:
      - Posición: se cancela solo el movimiento normal a la parcela (Y). El
        robot queda del lado de donde venía, a config.FARMING_PLOT_CONTACT_GAP
        del eje, y conserva su avance a lo largo de ella (X).
      - Velocidad lineal: se conserva.
      - Rumbo: a lo largo de la parcela, config.FARMING_PLOT_CONTACT_DEFLECTION
        hacia afuera de ella, en el sentido preferred_dir_x si no es 0 y si
        no en el sentido en que el robot avanzaba en X; más ruido
        (config.FARMING_PLOT_CONTACT_HEADING_NOISE_STD).

    Es un respaldo de la repulsión de parcela, que normalmente desvía al
    robot antes de que llegue a tocarla.

    Args:
        c_old:           Estado del robot antes del paso, shape (6,).
        c_new:           Estado tras integrar y resolver paredes, shape (6,).
        plots:           Parcelas {'x0', 'x1', 'y'}.
        preferred_dir_x: +1 / -1 sentido preferido en X; 0 = sin preferencia.

    Returns:
        Tupla (c_fixed, blocked): c_new sin cambios si no hubo cruce o el
        estado deslizado si lo hubo, y True si se bloqueó un cruce.
    """
    rx, ry = float(c_old[0]), float(c_old[1])
    tx, ty = float(c_new[0]), float(c_new[1])
    dy     = ty - ry

    if abs(dy) < _EPS:
        return c_new, False

    for plot in plots:
        yp = float(plot["y"])
        if (ry < yp < ty) or (ty < yp < ry):
            t       = (yp - ry) / dy
            x_cross = rx + t * (tx - rx)
            if float(plot["x0"]) <= x_cross <= float(plot["x1"]):
                side    = 1.0 if ry > yp else -1.0     # lado de donde venía el robot
                c_fixed = c_new.copy()
                c_fixed[1] = yp + side * config.FARMING_PLOT_CONTACT_GAP

                if preferred_dir_x != 0.0:
                    dir_x = 1.0 if preferred_dir_x > 0.0 else -1.0
                elif abs(tx - rx) > _EPS:
                    dir_x = 1.0 if tx > rx else -1.0
                else:
                    dir_x = 1.0 if math.cos(c_old[3]) >= 0.0 else -1.0

                defl      = config.FARMING_PLOT_CONTACT_DEFLECTION
                new_theta = math.atan2(side * math.sin(defl), dir_x * math.cos(defl))
                c_fixed[3] = wrap_angle(
                    new_theta
                    + np.random.normal(0.0, config.FARMING_PLOT_CONTACT_HEADING_NOISE_STD)
                )
                c_fixed[5] = 0.0
                return c_fixed, True

    return c_new, False


def _point_segment_distance(
    points: np.ndarray,
    seg_a:  np.ndarray,
    seg_b:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Distancia de uno o varios puntos al segmento seg_a -> seg_b (vectorizado).

    Si seg_a == seg_b es la distancia punto a punto.

    Args:
        points: Puntos [x, y], shape (K, 2).
        seg_a:  Extremo inicial del segmento [x, y].
        seg_b:  Extremo final del segmento [x, y].

    Returns:
        Tupla (dist, closest): distancias, shape (K,), y punto del segmento
        más cercano a cada punto, shape (K, 2).
    """
    ab    = np.asarray(seg_b, dtype=float) - np.asarray(seg_a, dtype=float)
    denom = float(ab[0] * ab[0] + ab[1] * ab[1])
    if denom < _EPS_SQ:
        u = np.zeros(len(points))
    else:
        u = np.clip(((points - seg_a) @ ab) / denom, 0.0, 1.0)
    closest = np.asarray(seg_a, dtype=float) + u[:, None] * ab
    diff    = points - closest
    dist    = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    return dist, closest


def _detect_target(
    robot_pos:   np.ndarray,
    robot_theta: float,
    target_pos:  np.ndarray,
    radius:      float,
    fov:         float,
    n_repulsion: int,
    n_walls:     int,
    plots:       list[dict],
) -> tuple[float, float, bool]:
    """Percepción de un objetivo puntual (objeto o nest) con un sensor del robot.

    La distancia y el ángulo percibidos tienen ruido gaussiano
    (config.FARMING_SENSOR_DISTANCE_NOISE_STD / ANGLE_NOISE_STD). Hay
    detección si la distancia percibida es <= radius, el objetivo está en
    el campo de visión del sensor (fov), no hay repulsión activa (vecinos en
    zona R, parcelas o paredes en contacto) y ninguna parcela se interpone en
    la línea robot-objetivo (oclusión con las posiciones reales). Con
    radius = 0 nunca hay detección.

    Args:
        robot_pos:   Posición [x, y] del robot (m).
        robot_theta: Rumbo del robot (rad).
        target_pos:  Posición [x, y] del objetivo (m).
        radius:      Radio de detección del objetivo, desde el centro del robot (m).
        fov:         Campo de visión del sensor, centrado en el rumbo (rad).
        n_repulsion: Vecinos en zona R más parcelas en contacto.
        n_walls:     Paredes en contacto.
        plots:       Parcelas {'x0', 'x1', 'y'} que ocluyen la vista.

    Returns:
        Tupla (distance, angle, detected): distancia percibida (m), ángulo
        percibido hacia el objetivo (rad) y True si el objetivo se detecta.
    """
    dx            = float(target_pos[0]) - robot_pos[0]
    dy            = float(target_pos[1]) - robot_pos[1]
    true_distance = math.sqrt(dx ** 2 + dy ** 2)
    true_angle    = wrap_angle(math.atan2(dy, dx))

    distance = max(
        0.0, true_distance + np.random.normal(0.0, config.FARMING_SENSOR_DISTANCE_NOISE_STD)
    )
    angle = wrap_angle(
        true_angle + np.random.normal(0.0, config.FARMING_SENSOR_ANGLE_NOISE_STD)
    )

    off_axis = min(wrap_angle(angle - robot_theta), wrap_angle(robot_theta - angle))

    detected = (
        radius > 0.0
        and off_axis < fov / 2
        and distance <= radius
        and n_repulsion == 0 and n_walls == 0
        and not segment_blocked_by_plots(robot_pos, target_pos, plots)
    )
    return distance, angle, detected


# ============================================================================
# Configuración de una corrida
# ============================================================================

@dataclass(frozen=True)
class _Setup:
    """Parámetros resueltos de una corrida (con los valores por defecto aplicados).

    Attributes:
        dyn:              Coeficientes del modelo dinámico del robot escalado.
        body_radius:      Radio del cuerpo del robot escalado (m).
        voltages:         Voltaje de régimen permanente de cada zona (V).
        area_limits:      Lado del área cuadrada (m).
        plots:            Parcelas {'x0', 'x1', 'y'}.
        nest_pos:         Posición [x, y] del nest (m).
        nest_ri:          Radio del sensor de nest de cada robot (m).
        nest_rs:          Radio de emisión del nest (m).
        nest_radius:      Alcance de detección del nest, r_i + r_s (m).
        radii_search:     Radios efectivos (R, O, A) sin carga, desde el centro (m).
        radii_deliver:    Radios efectivos (R, O, A) con carga, desde el centro (m).
        obj_sensor_range: Alcance del sensor de objetos desde el centro (m).
        max_iter:         Iteraciones máximas.
    """

    dyn:              DynamicsConstants
    body_radius:      float
    voltages:         dict
    area_limits:      float
    plots:            list
    nest_pos:         np.ndarray
    nest_ri:          float
    nest_rs:          float
    nest_radius:      float
    radii_search:     tuple[float, float, float]
    radii_deliver:    tuple[float, float, float]
    obj_sensor_range: float
    max_iter:         int


def _scaled_robot() -> tuple[DynamicsConstants, float]:
    """Construye el robot escalado a FARMING_ROBOT_SCALE.

    Masa ∝ escala³, inercia ∝ escala⁵, longitudes ∝ escala y el motor por
    similitud dinámica (MOTOR_Kl / escala⁵, ver config.FARMING_ROBOT_SCALE).

    Returns:
        Tupla (dyn, body_radius): coeficientes del modelo dinámico y radio
        del cuerpo (m).
    """
    scale = config.FARMING_ROBOT_SCALE
    dyn = DynamicsConstants(
        mass      = config.ROBOT_MASS      * scale ** 3,
        inertia   = config.ROBOT_INERTIA   * scale ** 5,
        d         = config.ROBOT_D         * scale,
        wheel_r   = config.ROBOT_WHEEL_R   * scale,
        wheel_sep = config.ROBOT_WHEEL_SEP * scale,
        motor_kl  = config.MOTOR_Kl / scale ** 5,
    )
    return dyn, config.ROBOT_BODY_RADIUS * scale


def _validate_inputs(
    n_objects:   int,
    n_plots:     int,
    individuals: int,
    radii:       dict[str, float],
    max_iter:    int,
) -> None:
    """Rechaza parámetros de corrida inválidos con un mensaje claro.

    Args:
        n_objects:   Número de objetos.
        n_plots:     Número de parcelas.
        individuals: Número de robots.
        radii:       Radios por nombre de parámetro (m).
        max_iter:    Iteraciones máximas.

    Raises:
        ValueError: Si algún conteo no es positivo, algún radio es negativo,
            los radios de una fase no cumplen r_r <= o_r <= a_r, o el número
            de parcelas excede el máximo que cabe en el área.
    """
    for name, value in (("n_objects", n_objects), ("n_plots", n_plots),
                        ("individuals", individuals), ("time_limit", max_iter)):
        if value < 1:
            raise ValueError(f"{name}={value} debe ser al menos 1.")
    for name, radius in radii.items():
        if radius < 0.0:
            raise ValueError(f"{name}={radius} no puede ser negativo.")
    for suffix in ("", "_deliver"):
        r_r, o_r, a_r = (radii[f"{z}{suffix}"] for z in ("r_r", "o_r", "a_r"))
        if not r_r <= o_r <= a_r:
            raise ValueError(
                f"Los radios r_r{suffix}={r_r}, o_r{suffix}={o_r} y a_r{suffix}={a_r} "
                f"deben cumplir r_r <= o_r <= a_r."
            )
    limit = max_plots()
    if n_plots > limit:
        raise ValueError(
            f"n_plots={n_plots} excede el maximo permitido ({limit}) "
            f"para area {config.FARMING_AREA_LIMITS}x{config.FARMING_AREA_LIMITS} m "
            f"con los parametros actuales."
        )
    if config.FARMING_PLOT_LENGTH + config.FARMING_PLOT_EAST_MARGIN >= config.FARMING_AREA_LIMITS:
        raise ValueError(
            f"FARMING_PLOT_LENGTH={config.FARMING_PLOT_LENGTH} m + FARMING_PLOT_EAST_MARGIN="
            f"{config.FARMING_PLOT_EAST_MARGIN} m no caben en un area de "
            f"{config.FARMING_AREA_LIMITS}x{config.FARMING_AREA_LIMITS} m: no quedaria pasillo "
            f"del lado oeste. Reduce FARMING_PLOT_LENGTH o FARMING_PLOT_EAST_MARGIN, o aumenta "
            f"FARMING_AREA_LIMITS en config.py."
        )


def _spawn_robots(individuals: int, r_rep_spawn: float, area_limits: float) -> np.ndarray:
    """Coloca a los robots en una columna vertical fuera del nest, mirando hacia +X.

    La columna nunca queda dentro del margen de la pared oeste. El
    espaciado ideal se reduce, si hace falta para que quepan todos, hasta
    el mínimo seguro (dos radios de repulsión: zonas de repulsión que
    apenas no se tocan).

    Args:
        individuals: Número de robots.
        r_rep_spawn: Radio de repulsión efectivo en la fase de búsqueda (m).
        area_limits: Lado del área cuadrada (m).

    Returns:
        Estado inicial del enjambre, shape (N, 6): [x, y, z, theta, v, omega].

    Raises:
        ValueError: Si los robots no caben en la columna ni con el espaciado mínimo.
    """
    spawn_x        = max(config.FARMING_SPAWN_LINE_X, r_rep_spawn + _SPAWN_WALL_CLEARANCE)
    safety_spacing = max(config.SPAWN_MIN_SEPARATION, 2.0 * r_rep_spawn)
    ideal_spacing  = max(config.SPAWN_MIN_SEPARATION,
                         config.FARMING_SPAWN_SPACING_FACTOR * r_rep_spawn)

    if individuals > 1:
        available_spacing = (area_limits - 2.0 * safety_spacing) / (individuals - 1)
    else:
        available_spacing = ideal_spacing
    spawn_spacing = min(ideal_spacing, available_spacing)

    if spawn_spacing < safety_spacing:
        max_individuals = int((area_limits - 2.0 * safety_spacing) / safety_spacing) + 1
        raise ValueError(
            f"individuals={individuals} no caben en la linea de spawn vertical "
            f"(x={spawn_x}) ni con el espaciado minimo seguro "
            f"({safety_spacing:.3f} m, 2x r_repulsion). Con "
            f"FARMING_AREA_LIMITS={area_limits} m caben como maximo "
            f"~{max_individuals} robots en linea. Reduce individuals o "
            f"aumenta FARMING_AREA_LIMITS."
        )
    if spawn_spacing < ideal_spacing:
        logger.warning(
            "Con individuals=%d, el espaciado de spawn se redujo de %.3f m (ideal) a "
            "%.3f m para que todos quepan en el area (sigue siendo seguro: >= %.3f m).",
            individuals, ideal_spacing, spawn_spacing, safety_spacing,
        )

    y_start = max(spawn_spacing, (area_limits - (individuals - 1) * spawn_spacing) / 2.0)

    C = np.zeros((individuals, 6))
    C[:, 0] = spawn_x
    C[:, 1] = y_start + np.arange(individuals) * spawn_spacing
    return C


# ============================================================================
# Estado del enjambre
# ============================================================================

@dataclass
class _SwarmState:
    """Estado mutable de todos los robots durante una corrida.

    Todos los arreglos tienen N filas, una por robot.

    Attributes:
        C:                  Estado dinámico [x, y, z, theta, v, omega], shape (N, 6).
        carrying:           Índice del objeto que carga cada robot; -1 = ninguno.
        compass_bias:       Sesgo fijo de la brújula de cada robot (rad).
        prev_state:         Zona activa de la iteración anterior (Zone; -1 al inicio).
        home_mode:          Fase del regreso con brújula: HOME_WEST o HOME_LEG.
        leg_count:          Tramos de vaivén iniciados en este viaje (el tramo k dura k·L).
        leg_dir:            Sentido del tramo actual: +1 norte, -1 sur.
        leg_left:           Iteraciones restantes del tramo actual.
        dir_explore:        Rumbo de exploración libre (rad).
        free_iters:         Iteraciones seguidas sin zona activa (hasta EXPLORE_FREE_ITERS).
        levy_run_remaining: Iteraciones restantes de la racha recta de Lévy.
        edge_follow_side:   Sentido de bordeo de parcela (+1 / -1); 0 = sin contacto.
        prev_target_theta:  Dirección deseada de la iteración anterior (rad).
        capped_streak:      Iteraciones seguidas con salto inestable de la dirección.
        escape_cooldown:    Iteraciones restantes de la válvula de escape.
        theta_last:         Rumbo al final de la iteración anterior (rad).
        step_rot_hist:      Giro absoluto de cada iteración en la ventana, shape (N, W).
        cum_rot:            Suma de step_rot_hist por robot (rad).
        pos_hist:           Posiciones recientes en la ventana, shape (N, W, 2).
        spin_flag:          Giro sostenido sin avance detectado.
        counters:           [iteraciones cargando, iteraciones buscando, objetos
                            recogidos] por robot, shape (N, 3).
    """

    C:                  np.ndarray
    carrying:           np.ndarray
    compass_bias:       np.ndarray
    prev_state:         np.ndarray
    home_mode:          np.ndarray
    leg_count:          np.ndarray
    leg_dir:            np.ndarray
    leg_left:           np.ndarray
    dir_explore:        np.ndarray
    free_iters:         np.ndarray
    levy_run_remaining: np.ndarray
    edge_follow_side:   np.ndarray
    prev_target_theta:  np.ndarray
    capped_streak:      np.ndarray
    escape_cooldown:    np.ndarray
    theta_last:         np.ndarray
    step_rot_hist:      np.ndarray
    cum_rot:            np.ndarray
    pos_hist:           np.ndarray
    spin_flag:          np.ndarray
    counters:           np.ndarray

    @classmethod
    def create(cls, C: np.ndarray, rng: np.random.Generator) -> "_SwarmState":
        """Inicializa el estado a partir de las posiciones de arranque.

        Args:
            C:   Estado dinámico inicial, shape (N, 6).
            rng: Generador aleatorio (sortea el sesgo de brújula de cada robot).

        Returns:
            Estado inicial del enjambre.
        """
        n      = C.shape[0]
        window = config.FARMING_SPIN_WINDOW
        return cls(
            C                  = C,
            carrying           = np.full(n, -1, dtype=int),
            compass_bias       = rng.normal(0.0, config.FARMING_COMPASS_ERROR_STD, n),
            prev_state         = np.full(n, -1, dtype=int),
            home_mode          = np.full(n, HOME_WEST, dtype=int),
            leg_count          = np.zeros(n, dtype=int),
            leg_dir            = np.zeros(n, dtype=int),
            leg_left           = np.zeros(n, dtype=int),
            dir_explore        = C[:, 3].copy(),
            free_iters         = np.zeros(n, dtype=int),
            levy_run_remaining = np.zeros(n, dtype=int),
            edge_follow_side   = np.zeros(n, dtype=int),
            prev_target_theta  = C[:, 3].copy(),
            capped_streak      = np.zeros(n, dtype=int),
            escape_cooldown    = np.zeros(n, dtype=int),
            theta_last         = C[:, 3].copy(),
            step_rot_hist      = np.zeros((n, window)),
            cum_rot            = np.zeros(n),
            pos_hist           = np.repeat(C[:, :2][:, np.newaxis, :], window, axis=1),
            spin_flag          = np.zeros(n, dtype=bool),
            counters           = np.zeros((n, 3)),
        )


@dataclass
class _Objects:
    """Estado de los objetos durante una corrida.

    Attributes:
        pos:       Posiciones [x, y] de los objetos, shape (O, 2).
        on_ground: True si el objeto está libre en el suelo (ni cargado ni entregado).
        delivered: True si el objeto ya se depositó en el nest.
        remaining: Objetos aún sin entregar.
    """

    pos:       np.ndarray
    on_ground: np.ndarray
    delivered: np.ndarray
    remaining: int


# ============================================================================
# Decisión de cada robot
# ============================================================================

def _select_target(
    i:         int,
    swarm:     _SwarmState,
    has_load:  bool,
    avail_idx: np.ndarray,
    avail_pos: np.ndarray,
    objects:   _Objects,
    setup:     _Setup,
) -> tuple[np.ndarray, float, float]:
    """Elige el objetivo de la influencia del robot i y el sensor que lo percibe.

    Sin carga, con el sensor de objetos: el objeto más cercano dentro de su
    alcance y de su cono frontal, con línea de vista libre, recalculado en
    cada iteración; sin objeto visible el sensor no tiene objetivo (radio 0).
    Con carga, con el sensor de nest (alcance r_i + r_s).

    Args:
        i:         Índice del robot.
        swarm:     Estado del enjambre.
        has_load:  True si el robot carga un objeto.
        avail_idx: Índices de los objetos en el suelo al inicio de la iteración.
        avail_pos: Posiciones de esos objetos, shape (K, 2).
        objects:   Estado de los objetos.
        setup:     Parámetros de la corrida.

    Returns:
        Tupla (target_pos, target_radius, sensor_fov): posición del objetivo
        (m), alcance de detección desde el centro del robot (m) y campo de
        visión del sensor (rad).
    """
    if has_load:
        return setup.nest_pos, setup.nest_radius, config.FARMING_FOV_NEST_SENSOR

    C = swarm.C
    fov = config.FARMING_FOV_OBJECT_SENSOR
    if avail_idx.size:
        diff  = avail_pos - C[i, :2]
        d_all = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
        off   = np.abs(angle_diff(np.arctan2(diff[:, 1], diff[:, 0]), C[i, 3]))
        near  = np.flatnonzero((d_all <= setup.obj_sensor_range) & (off < fov / 2))
        for k in near[np.argsort(d_all[near], kind="stable")]:
            o = int(avail_idx[k])
            if not segment_blocked_by_plots(C[i, :2], objects.pos[o], setup.plots):
                return objects.pos[o], setup.obj_sensor_range, fov
    return C[i, :2], 0.0, fov


def _repulsion_direction(
    i:        int,
    swarm:    _SwarmState,
    sources:  list,
    plot_pts: list,
    has_load: bool,
    setup:    _Setup,
) -> tuple[float, float]:
    """Dirección unitaria de repulsión del robot i, con bordeo de parcelas.

    Si hay parcelas en contacto, la repulsión se mezcla con una de sus dos
    tangentes. El sentido se elige al entrar en contacto y se conserva
    mientras dure: con carga, hacia el lado del nest en X (con el sesgo de
    la brújula); sin carga, el más alineado con el rumbo actual.

    Args:
        i:        Índice del robot.
        swarm:    Estado del enjambre (actualiza edge_follow_side).
        sources:  Fuentes repulsivas: vecinos en zona R, puntos de pared y de parcela.
        plot_pts: Puntos de parcela en contacto.
        has_load: True si el robot carga un objeto.
        setup:    Parámetros de la corrida.

    Returns:
        Vector unitario (vx, vy) de la zona de repulsión.
    """
    C = swarm.C
    rvx, rvy = repulsion_vector(C[i, :2], sources)
    r_norm   = max(math.sqrt(rvx ** 2 + rvy ** 2), _EPS)
    rvx, rvy = rvx / r_norm, rvy / r_norm

    if not plot_pts:
        return rvx, rvy

    tx1, ty1 = -rvy, rvx
    tx2, ty2 =  rvy, -rvx

    if swarm.edge_follow_side[i] == 0:
        if has_load:
            base_bearing = 0.0 if setup.nest_pos[0] - C[i, 0] >= 0.0 else math.pi
            ref_angle    = base_bearing + swarm.compass_bias[i]
        else:
            ref_angle = C[i, 3]
        ref_x, ref_y = math.cos(ref_angle), math.sin(ref_angle)
        d1 = tx1 * ref_x + ty1 * ref_y
        d2 = tx2 * ref_x + ty2 * ref_y
        swarm.edge_follow_side[i] = 1 if d1 >= d2 else -1

    tx, ty = (tx1, ty1) if swarm.edge_follow_side[i] == 1 else (tx2, ty2)

    ew  = config.FARMING_EDGE_FOLLOW_WEIGHT
    bx  = (1.0 - ew) * rvx + ew * tx
    by  = (1.0 - ew) * rvy + ew * ty
    b_n = max(math.sqrt(bx ** 2 + by ** 2), _EPS)
    return bx / b_n, by / b_n


def _unit_sum(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Suma de vectores normalizada a norma 1 (protegida contra norma 0).

    Args:
        xs: Componentes X.
        ys: Componentes Y.

    Returns:
        Vector unitario (vx, vy).
    """
    vx, vy = sum(xs), sum(ys)
    norm = max(math.sqrt(vx ** 2 + vy ** 2), _EPS)
    return vx / norm, vy / norm


def _dominant_zone(active: dict) -> Zone:
    """Zona dominante para el reporte: R > I > O > A > exploración libre.

    Args:
        active: Zonas activas {'R', 'O', 'A', 'I'} del robot.

    Returns:
        Código de la zona dominante.
    """
    for key, zone in (("R", Zone.REPULSION), ("I", Zone.INFLUENCE),
                      ("O", Zone.ORIENTATION), ("A", Zone.ATTRACTION)):
        if key in active:
            return zone
    return Zone.FREE


def _desired_heading(
    i:        int,
    swarm:    _SwarmState,
    active:   dict,
    has_load: bool,
    rng:      np.random.Generator,
) -> float:
    """Dirección deseada del robot i.

    Con alguna zona activa, la composición RAOI (y, si el robot venía de
    explorar, se reinicia el vuelo de Lévy). Sin zonas activas: con carga,
    la brújula (rumbo oeste hasta tocar la cerca y luego tramos norte-sur
    de la búsqueda en vaivén, con sesgo fijo y ruido por lectura); sin
    carga, el vuelo de Lévy (rachas rectas de duración de cola pesada).

    Args:
        i:        Índice del robot.
        swarm:    Estado del enjambre (actualiza exploración y búsqueda).
        active:   Zonas activas del robot.
        has_load: True si el robot carga un objeto.
        rng:      Generador aleatorio.

    Returns:
        Dirección deseada (rad).
    """
    C = swarm.C
    if active:
        target_theta = combined_direction(C[i, 3], active, config.FARMING_RAOI_WEIGHTS)
        if swarm.prev_state[i] == Zone.FREE:
            swarm.free_iters[i]         = 0
            swarm.levy_run_remaining[i] = 0
            swarm.dir_explore[i] = wrap_angle(
                C[i, 3] + np.random.normal(0.0, config.DIREXP_RESET_NOISE)
            )
        return target_theta

    if has_load:
        if swarm.home_mode[i] == HOME_LEG:
            heading = swarm.leg_dir[i] * math.pi / 2
            swarm.leg_left[i] -= 1
            if swarm.leg_left[i] <= 0:
                swarm.home_mode[i] = HOME_WEST      # fin del tramo: regresa a la cerca
        else:
            heading = config.FARMING_COMPASS_HEADING
        return wrap_angle(
            heading + swarm.compass_bias[i]
            + np.random.normal(0.0, config.FARMING_COMPASS_NOISE_STD)
        )

    if swarm.free_iters[i] < config.EXPLORE_FREE_ITERS:
        swarm.free_iters[i] += 1
    elif swarm.levy_run_remaining[i] <= 0:
        swarm.levy_run_remaining[i] = int(np.clip(
            (rng.pareto(config.FARMING_LEVY_EXPONENT) + 1.0) * config.FARMING_LEVY_MIN_RUN,
            config.FARMING_LEVY_MIN_RUN,
            config.FARMING_LEVY_MAX_RUN,
        ))
        swarm.dir_explore[i] = wrap_angle(rng.uniform(0.0, 2 * math.pi))
    else:
        swarm.levy_run_remaining[i] -= 1
        swarm.dir_explore[i] = wrap_angle(
            swarm.dir_explore[i] + np.random.normal(0.0, config.FARMING_LEVY_TURN_NOISE_STD)
        )
    return swarm.dir_explore[i]


def _reset_spin_window(swarm: _SwarmState, i: int) -> None:
    """Vacía la ventana de giro sostenido del robot i (giro intencional o válvula activada)."""
    swarm.cum_rot[i]          = 0.0
    swarm.step_rot_hist[i, :] = 0.0


def _adjust_heading(
    i:              int,
    t:              int,
    swarm:          _SwarmState,
    target_theta:   float,
    final_approach: bool,
) -> None:
    """Ajusta el rumbo del robot i hacia la dirección deseada (de forma cinemática).

    Por orden de prioridad:
      1. Aproximación final: estímulo real a menos de
         FARMING_FINAL_APPROACH_RADIUS -> el robot gira en su lugar hacia la
         dirección deseada.
      2. Válvula de escape en curso -> sigue derecho.
      3. Giro sostenido sin avance detectado -> activa la válvula.
      4. Salto inestable de la dirección deseada -> giro limitado a
         FARMING_TURN_CAP; tras FARMING_TURN_STREAK_LIMIT saltos seguidos,
         activa la válvula.
      5. Caso normal -> promedio 50/50 entre rumbo actual y deseado.

    Args:
        i:              Índice del robot.
        t:              Iteración actual.
        swarm:          Estado del enjambre (actualiza C[i, 3] y la ejecución del giro).
        target_theta:   Dirección deseada (rad).
        final_approach: True si el robot está en aproximación final.
    """
    C = swarm.C
    desired_jump = abs(angle_diff(target_theta, swarm.prev_target_theta[i])) if t > 0 else 0.0

    if final_approach:
        C[i, 3]                   = wrap_angle(target_theta)
        swarm.escape_cooldown[i]  = 0
        swarm.capped_streak[i]    = 0
        _reset_spin_window(swarm, i)
    elif swarm.escape_cooldown[i] > 0:
        swarm.escape_cooldown[i] -= 1
    elif swarm.spin_flag[i]:
        swarm.escape_cooldown[i]  = config.FARMING_TURN_COOLDOWN
        swarm.capped_streak[i]    = 0
        _reset_spin_window(swarm, i)
    elif desired_jump > config.FARMING_TURN_JUMP_THRESHOLD:
        swarm.capped_streak[i] += 1
        if swarm.capped_streak[i] > config.FARMING_TURN_STREAK_LIMIT:
            swarm.escape_cooldown[i] = config.FARMING_TURN_COOLDOWN
            swarm.capped_streak[i]   = 0
            _reset_spin_window(swarm, i)
        else:
            turn = angle_diff(target_theta, C[i, 3])
            turn = max(-config.FARMING_TURN_CAP, min(config.FARMING_TURN_CAP, turn))
            C[i, 3] = wrap_angle(C[i, 3] + turn)
    else:
        swarm.capped_streak[i] = 0
        x_mean = 0.5 * math.cos(C[i, 3]) + 0.5 * math.cos(target_theta)
        y_mean = 0.5 * math.sin(C[i, 3]) + 0.5 * math.sin(target_theta)
        C[i, 3] = wrap_angle(math.atan2(y_mean, x_mean))


def _decide(
    i:         int,
    t:         int,
    swarm:     _SwarmState,
    objects:   _Objects,
    avail_idx: np.ndarray,
    avail_pos: np.ndarray,
    setup:     _Setup,
    rng:       np.random.Generator,
) -> tuple[np.ndarray, Zone]:
    """Percepción, reglas RAOI y ajuste de rumbo del robot i en la iteración t.

    Args:
        i:         Índice del robot.
        t:         Iteración actual.
        swarm:     Estado del enjambre.
        objects:   Estado de los objetos.
        avail_idx: Índices de los objetos en el suelo al inicio de la iteración.
        avail_pos: Posiciones de esos objetos, shape (K, 2).
        setup:     Parámetros de la corrida.
        rng:       Generador aleatorio.

    Returns:
        Tupla (voltages, zone): voltajes [u_r, u_l] de las ruedas y zona
        dominante del robot en esta iteración.
    """
    C        = swarm.C
    noise    = np.random.normal(0.0, config.FARMING_ACTUATOR_NOISE_STD)
    has_load = swarm.carrying[i] >= 0

    # Percepción del entorno y de los vecinos.
    r_repulsion, r_orientation, r_attraction = (
        setup.radii_deliver if has_load else setup.radii_search
    )
    walls    = detect_walls(C[i, :2], r_repulsion, setup.area_limits)
    plot_pts = detect_plots(C[i, :2], setup.plots, config.FARMING_PLOT_REPULSION)
    if not plot_pts:
        swarm.edge_follow_side[i] = 0
    neighbors = detect_neighbors(
        i, C, r_repulsion, r_orientation, r_attraction,
        config.RAOI_FOV, len(walls) + len(plot_pts), setup.plots,
    )
    n_rep_total = neighbors["n_rep"] + len(plot_pts)

    # Influencia.
    target_pos, target_radius, sensor_fov = _select_target(
        i, swarm, has_load, avail_idx, avail_pos, objects, setup
    )
    inf_dist, inf_angle, inf_detected = _detect_target(
        C[i, :2], C[i, 3], target_pos, target_radius, sensor_fov,
        n_rep_total, len(walls), setup.plots,
    )

    # Vectores RAOI.
    active: dict[str, tuple[float, float]] = {}
    sources = neighbors["rep_neighbors"] + walls + plot_pts
    if sources:
        active["R"] = _repulsion_direction(i, swarm, sources, plot_pts, has_load, setup)
    if neighbors["n_ori"] > 0:
        active["O"] = _unit_sum(neighbors["ox"], neighbors["oy"])
    if neighbors["n_att"] > 0:
        active["A"] = _unit_sum(neighbors["ax"], neighbors["ay"])
    if inf_detected:
        active["I"] = (math.cos(inf_angle), math.sin(inf_angle))

    zone = _dominant_zone(active)

    # Dirección deseada y ajuste del rumbo.
    target_theta   = _desired_heading(i, swarm, active, has_load, rng)
    final_approach = "I" in active and inf_dist <= config.FARMING_FINAL_APPROACH_RADIUS
    _adjust_heading(i, t, swarm, target_theta, final_approach)

    swarm.prev_target_theta[i] = target_theta
    swarm.prev_state[i]        = zone

    return select_voltage(active, inf_dist, target_radius, noise, setup.voltages), zone


# ============================================================================
# Dinámica, contacto, recolección y depósito
# ============================================================================

def _apply_motion(
    swarm:            _SwarmState,
    desired_voltages: np.ndarray,
    setup:            _Setup,
    rng:              np.random.Generator,
) -> np.ndarray:
    """Integra la dinámica y resuelve el contacto con paredes y parcelas.

    Además avanza la búsqueda en vaivén de los robots cargados que tocan
    una pared (ver _advance_search_leg).

    Args:
        swarm:            Estado del enjambre (actualiza C y el regreso con brújula).
        desired_voltages: Voltajes [u_r, u_l] de cada robot, shape (N, 2).
        setup:            Parámetros de la corrida.
        rng:              Generador aleatorio.

    Returns:
        Posiciones al inicio del paso, shape (N, 2).
    """
    C          = swarm.C
    area       = setup.area_limits
    pos_before = C[:, :2].copy()
    C_int      = dynamics_step(C, desired_voltages, setup.dyn)
    for i in range(C.shape[0]):
        has_load = swarm.carrying[i] >= 0
        c_old    = C[i].copy()
        margin   = setup.radii_deliver[0] if has_load else setup.radii_search[0]
        c_new, bounced = resolve_walls(c_old, C_int[i], margin, area, config.FARMING_MAX_SPEED)

        # Con carga, el robot recorre la parcela hacia el lado del nest.
        pref_x = (1.0 if setup.nest_pos[0] >= c_old[0] else -1.0) if has_load else 0.0
        c_new, plot_blocked = _resolve_plot_collision(c_old, c_new, setup.plots, pref_x)
        C[i] = c_new
        if bounced or plot_blocked:
            swarm.dir_explore[i] = C[i, 3]

        if has_load and bounced:
            _advance_search_leg(swarm, i, margin + config.WALL_CONTACT_GAP + _EPS, area, rng)
    return pos_before


def _advance_search_leg(
    swarm: _SwarmState,
    i:     int,
    inner: float,
    area:  float,
    rng:   np.random.Generator,
) -> None:
    """Avanza la búsqueda en vaivén del robot cargado i tras un contacto con pared.

    Tocar la cerca oeste inicia el siguiente tramo (sentido al azar en el
    primero, alternado después; el tramo k dura k · FARMING_SEARCH_LEG_ITERS)
    y tocar la cerca norte o sur en el sentido del tramo lo termina.

    Args:
        swarm: Estado del enjambre (actualiza el regreso con brújula).
        i:     Índice del robot.
        inner: Distancia a la pared por debajo de la cual se considera contacto (m).
        area:  Lado del área cuadrada (m).
        rng:   Generador aleatorio.
    """
    x, y = swarm.C[i, 0], swarm.C[i, 1]
    if swarm.home_mode[i] == HOME_WEST and x <= inner:
        swarm.leg_count[i] += 1
        if swarm.leg_count[i] == 1:
            swarm.leg_dir[i] = 1 if rng.random() < 0.5 else -1
        else:
            swarm.leg_dir[i] = -swarm.leg_dir[i]
        swarm.leg_left[i]  = swarm.leg_count[i] * config.FARMING_SEARCH_LEG_ITERS
        swarm.home_mode[i] = HOME_LEG
    elif swarm.home_mode[i] == HOME_LEG and (
        (swarm.leg_dir[i] > 0 and y >= area - inner)
        or (swarm.leg_dir[i] < 0 and y <= inner)
    ):
        swarm.home_mode[i] = HOME_WEST


def _update_spin_window(swarm: _SwarmState, t: int) -> None:
    """Registra el giro real y la posición de la iteración en la ventana de giro sostenido.

    La condición de giro sostenido (spin_flag) se evalúa aquí y se usa al
    ajustar el rumbo en la iteración siguiente.

    Args:
        swarm: Estado del enjambre.
        t:     Iteración actual.
    """
    C      = swarm.C
    window = config.FARMING_SPIN_WINDOW
    step_rot         = np.abs(angle_diff(C[:, 3], swarm.theta_last))
    swarm.theta_last = C[:, 3].copy()

    slot = t % window
    swarm.cum_rot -= swarm.step_rot_hist[:, slot]
    swarm.cum_rot += step_rot
    swarm.step_rot_hist[:, slot] = step_rot
    swarm.pos_hist[:, slot, :]   = C[:, :2]

    centroid = swarm.pos_hist.mean(axis=1)
    spread   = np.sqrt(((swarm.pos_hist - centroid[:, None, :]) ** 2).sum(axis=2)).max(axis=1)
    swarm.spin_flag = (
        (t >= window)
        & (swarm.cum_rot > config.FARMING_SPIN_ROTATION_THRESHOLD)
        & (spread < config.FARMING_SPIN_MAX_SPREAD)
    )


def _pick_and_deposit(
    swarm:      _SwarmState,
    objects:    _Objects,
    pos_before: np.ndarray,
    setup:      _Setup,
) -> None:
    """Recolección y depósito sobre el tramo recorrido en la iteración.

    Con config.FARMING_SWEPT_CONTACT el contacto se evalúa sobre el tramo
    pos_before -> posición final; si no, solo en la posición final. Un
    robot cargado deposita si el tramo pasa a menos de
    FARMING_DEPOSIT_RADIUS del nest (el objeto queda en el punto del tramo
    más cercano al nest); uno sin carga recoge el objeto en el suelo más
    cercano al tramo si está a menos de FARMING_PICK_RADIUS.

    Args:
        swarm:      Estado del enjambre (actualiza carga, búsqueda y contadores).
        objects:    Estado de los objetos.
        pos_before: Posiciones al inicio del paso, shape (N, 2).
        setup:      Parámetros de la corrida.
    """
    C = swarm.C
    for i in range(C.shape[0]):
        seg_a = pos_before[i] if config.FARMING_SWEPT_CONTACT else C[i, :2]
        seg_b = C[i, :2]

        if swarm.carrying[i] >= 0:
            objects.pos[swarm.carrying[i]] = C[i, :2].copy()

            d_nest, drop = _point_segment_distance(setup.nest_pos[None, :], seg_a, seg_b)
            if d_nest[0] <= config.FARMING_DEPOSIT_RADIUS:
                o = swarm.carrying[i]
                objects.pos[o]       = drop[0]
                objects.delivered[o] = True
                objects.remaining   -= 1
                swarm.carrying[i]    = -1
                swarm.home_mode[i]   = HOME_WEST
                swarm.leg_count[i]   = 0
        else:
            free = np.flatnonzero(objects.on_ground)
            if free.size:
                d_obj, _ = _point_segment_distance(objects.pos[free], seg_a, seg_b)
                k = int(np.argmin(d_obj))
                if d_obj[k] <= config.FARMING_PICK_RADIUS:
                    o = int(free[k])
                    swarm.carrying[i]     = o
                    objects.on_ground[o]  = False
                    swarm.counters[i, 2] += 1
                    swarm.home_mode[i]    = HOME_WEST     # nuevo viaje: búsqueda desde cero
                    swarm.leg_count[i]    = 0

        if swarm.carrying[i] >= 0:
            swarm.counters[i, 0] += 1


# ============================================================================
# Simulación
# ============================================================================

def run(
    n_objects:     int,
    n_plots:       int,
    individuals:   int,
    r_r:           float,
    o_r:           float,
    a_r:           float,
    animation:     bool,
    seed:          int | None = None,
    r_r_deliver:   float | None = None,
    o_r_deliver:   float | None = None,
    a_r_deliver:   float | None = None,
    nest_ri:       float | None = None,
    nest_rs:       float | None = None,
    obj_r:         float | None = None,
    time_limit:    int | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    output_dir:    str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ejecuta la simulación de la tarea de farming con el modelo RAOI.

    Los radios RAOI se miden desde el borde del robot (el radio efectivo es
    el radio del cuerpo más el radio dado), igual que el sensor de objetos;
    el sensor de nest y el emisor se miden desde el centro.

    Args:
        n_objects:   Número de objetos a recolectar.
        n_plots:     Número de parcelas.
        individuals: Número de robots.
        r_r:         Radio de repulsión en la fase de búsqueda, sin objeto (m).
        o_r:         Radio de orientación en la fase de búsqueda (m).
        a_r:         Radio de atracción en la fase de búsqueda (m).
        animation:   Si True, reproduce o graba la animación al terminar.
        seed:        Semilla aleatoria. None -> config.SEED.
        r_r_deliver: Radio de repulsión en la fase de entrega, cargando un
                     objeto (m). None -> r_r.
        o_r_deliver: Radio de orientación en la fase de entrega (m). None -> o_r.
        a_r_deliver: Radio de atracción en la fase de entrega (m). None -> a_r.
        nest_ri:     Radio del sensor de nest del robot (m). None -> config.FARMING_NEST_RI.
        nest_rs:     Radio de emisión del nest (m); el nest se detecta dentro de
                     nest_ri + nest_rs. None -> config.FARMING_NEST_RS.
        obj_r:       Radio r_d del sensor de objetos, desde el borde del robot (m).
                     None -> config.FARMING_OBJECT_SENSOR_RADIUS.
        time_limit:  Iteraciones máximas. None -> config.FARMING_DEFAULT_MAX_ITER.
        progress_callback: f(t, max_iter, n_delivered), llamada al final de cada
                     iteración (t empieza en 0).
        output_dir:  Carpeta donde se graba el video de la animación
                     (config.VIDEO_FILENAME) si config.SAVE_VIDEO es True.
                     None -> carpeta actual.

    Returns:
        Tupla (report, objects_report, metrics):
          - report: estado del enjambre por iteración, shape (T, N, 8):
            [x, y, z, theta (rad), theta (grados), v, omega, zona activa].
          - objects_report: posiciones de los objetos por iteración, shape (T, O, 2).
          - metrics: funciones objetivo f1–f4 (ver metrics.farming_metrics).

    Raises:
        ValueError: Si los parámetros son inválidos (ver _validate_inputs) o
            los robots no caben en la columna de arranque.
    """
    # ── Parámetros de la corrida ──────────────────────────────────────────────
    radii = {
        "r_r": r_r, "o_r": o_r, "a_r": a_r,
        "r_r_deliver": r_r_deliver if r_r_deliver is not None else r_r,
        "o_r_deliver": o_r_deliver if o_r_deliver is not None else o_r,
        "a_r_deliver": a_r_deliver if a_r_deliver is not None else a_r,
        "nest_ri": nest_ri if nest_ri is not None else config.FARMING_NEST_RI,
        "nest_rs": nest_rs if nest_rs is not None else config.FARMING_NEST_RS,
        "obj_r":   obj_r if obj_r is not None else config.FARMING_OBJECT_SENSOR_RADIUS,
    }
    max_iter = int(time_limit) if time_limit is not None else config.FARMING_DEFAULT_MAX_ITER
    _validate_inputs(n_objects, n_plots, individuals, radii, max_iter)

    _seed = seed if seed is not None else config.SEED
    np.random.seed(_seed)
    rng = np.random.default_rng(_seed)

    dyn, body_radius = _scaled_robot()
    area_limits = config.FARMING_AREA_LIMITS
    setup = _Setup(
        dyn         = dyn,
        body_radius = body_radius,
        # Voltaje de régimen permanente de cada zona: v = Ks · r · u.
        voltages    = {
            zone: speed / (config.MOTOR_Ks * dyn.r)
            for zone, speed in config.FARMING_ZONE_SPEEDS.items()
        },
        area_limits = area_limits,
        plots       = _build_plots_symmetric(
            n_plots, config.FARMING_PLOT_LENGTH, area_limits,
            config.FARMING_PLOT_SEPARATION, config.FARMING_PLOT_EAST_MARGIN,
        ),
        nest_pos    = np.array(config.FARMING_NEST_POSITION, dtype=float),
        nest_ri     = radii["nest_ri"],
        nest_rs     = radii["nest_rs"],
        nest_radius = radii["nest_ri"] + radii["nest_rs"],
        radii_search = (
            body_radius + radii["r_r"], body_radius + radii["o_r"], body_radius + radii["a_r"]
        ),
        radii_deliver = (
            body_radius + radii["r_r_deliver"],
            body_radius + radii["o_r_deliver"],
            body_radius + radii["a_r_deliver"],
        ),
        obj_sensor_range = body_radius + radii["obj_r"],
        max_iter         = max_iter,
    )

    # ── Objetos y robots ──────────────────────────────────────────────────────
    objects = _Objects(
        pos=_spawn_objects_farming(
            n_objects, setup.plots, config.FARMING_STRIP_WIDTH,
            config.FARMING_PLOT_REPULSION, area_limits, rng,
        ),
        on_ground=np.ones(n_objects, dtype=bool),
        delivered=np.zeros(n_objects, dtype=bool),
        remaining=n_objects,
    )
    swarm = _SwarmState.create(
        _spawn_robots(individuals, setup.radii_search[0], area_limits), rng
    )

    # ── Búferes del reporte ───────────────────────────────────────────────────
    report_buf          = np.zeros((max_iter, individuals, 8))
    obj_report_buf      = np.zeros((max_iter, n_objects, 2))
    carrying_report_buf = np.full((max_iter, individuals), -1, dtype=int)

    # ── Ciclo principal ───────────────────────────────────────────────────────
    t = 0
    while t < max_iter and objects.remaining > 0:
        desired_voltages = np.zeros((individuals, 2))
        zone_state       = np.zeros(individuals)   # zona activa de cada robot en esta iteración

        avail_idx = np.flatnonzero(objects.on_ground)   # objetos en el suelo al inicio
        avail_pos = objects.pos[avail_idx]

        for i in range(individuals):
            desired_voltages[i], zone_state[i] = _decide(
                i, t, swarm, objects, avail_idx, avail_pos, setup, rng
            )

        pos_before = _apply_motion(swarm, desired_voltages, setup, rng)
        _update_spin_window(swarm, t)
        _pick_and_deposit(swarm, objects, pos_before, setup)

        C = swarm.C
        report_buf[t, :, 0:4] = C[:, 0:4]
        report_buf[t, :, 4]   = np.degrees(C[:, 3])
        report_buf[t, :, 5:7] = C[:, 4:6]
        report_buf[t, :, 7]   = zone_state
        obj_report_buf[t]      = objects.pos
        carrying_report_buf[t] = swarm.carrying

        if progress_callback is not None:
            progress_callback(t, max_iter, int(np.sum(objects.delivered)))
        t += 1

    # ── Resultados ────────────────────────────────────────────────────────────
    report          = report_buf[:t]
    objects_report  = obj_report_buf[:t]

    swarm.counters[:, 1] = t - swarm.counters[:, 0]
    result = farming_metrics(
        robot_counters = swarm.counters,
        n_objects      = n_objects,
        delivered      = objects.delivered,
        sensor_radius  = setup.nest_ri,
        emitter_radius = setup.nest_rs,
    )

    if animation:
        video_path = None
        if config.SAVE_VIDEO:
            video_path = Path(output_dir if output_dir is not None else ".") / config.VIDEO_FILENAME
        _animate(report, objects_report, carrying_report_buf[:t], setup, video_path)

    return report, objects_report, result


def _animate(
    report:          np.ndarray,
    objects_report:  np.ndarray,
    carrying_report: np.ndarray,
    setup:           _Setup,
    video_path:      Path | None,
) -> None:
    """Reproduce o graba la animación de una corrida terminada.

    La visualización (pygame y OpenCV) se importa aquí para que las
    corridas sin animación no dependan de ella.

    Args:
        report:          Estado del enjambre por iteración, shape (T, N, 8).
        objects_report:  Posiciones de los objetos, shape (T, O, 2).
        carrying_report: Objeto cargado por cada robot (-1 = ninguno), shape (T, N).
        setup:           Parámetros de la corrida.
        video_path:      Video mp4 de salida; None = ventana Pygame.
    """
    from . import visualization  # noqa: PLC0415  (import diferido: dependencia opcional)

    env = {
        "area_limits":           setup.area_limits,
        "nest_position":         list(setup.nest_pos),
        "nest_ri":               setup.nest_ri,
        "nest_rs":               setup.nest_rs,
        "obj_sensor_range":      setup.obj_sensor_range,
        "obj_sensor_fov":        config.FARMING_FOV_OBJECT_SENSOR,
        "nest_icon_radius":      config.FARMING_NEST_ICON_RADIUS,
        "deposit_radius":        config.FARMING_DEPOSIT_RADIUS,
        "plots":                 setup.plots,
        "plot_repulsion":        config.FARMING_PLOT_REPULSION,
        "strip_width":           config.FARMING_STRIP_WIDTH,
        "strip_gap":             config.FARMING_STRIP_GAP,
        "plot_visual_thickness": config.FARMING_PLOT_VISUAL_THICKNESS,
        "raoi_radii_search":     setup.radii_search,
        "raoi_radii_deliver":    setup.radii_deliver,
    }
    logger.info("Simulacion terminada. Iniciando la animacion...")
    visualization.animate_farming(
        report          = report,
        objects_report  = objects_report,
        carrying_report = carrying_report,
        env             = env,
        interval        = config.FARMING_ANIMATION_INTERVAL,
        show_zones      = config.SHOW_ZONES,
        show_trail      = config.SHOW_TRAIL,
        trail_length    = config.TRAIL_LENGTH,
        show_sensor     = config.SHOW_SENSOR_RADIUS,
        save_path       = str(video_path) if video_path is not None else None,
        screen_size     = config.SCREEN_SIZE,
    )


# ============================================================================
# Ejecución con consola
# ============================================================================

def _make_run_dir(root: str | Path, kind: str) -> Path:
    """Crea la carpeta de resultados de una corrida, con fecha y hora.

    Args:
        root: Carpeta raíz de resultados (por ejemplo config.RESULTS_DIR).
        kind: Tipo de corrida: "single" o "stats".

    Returns:
        Carpeta creada, por ejemplo results/2026-10-19_14-30-05_single. Si ya
        existe una con ese nombre se agrega un sufijo _2, _3, ...
    """
    base = Path(root) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{kind}"
    run_dir, n = base, 1
    while run_dir.exists():
        n += 1
        run_dir = base.with_name(f"{base.name}_{n}")
    run_dir.mkdir(parents=True)
    return run_dir


_REPORT_FILE_DESCRIPTION = (
    "farming_report.npy: swarm state per iteration, shape (T, N, 8): "
    "[x, y, z, theta (rad), theta (deg), v, omega, active zone]"
)
_OBJECTS_FILE_DESCRIPTION = (
    "farming_objects_report.npy: object positions per iteration, shape (T, O, 2)"
)


def _collect_params(
    n_objects:   int | None,
    n_plots:     int | None,
    individuals: int | None,
    r_r:         float | None,
    o_r:         float | None,
    a_r:         float | None,
    r_r_deliver: float | None,
    o_r_deliver: float | None,
    a_r_deliver: float | None,
    nest_ri:     float | None,
    nest_rs:     float | None,
    obj_r:       float | None,
    time_limit:  int | None,
) -> dict:
    """Agrupa los parámetros de corrida y pide por consola los que falten.

    Args:
        n_objects:   Número de objetos.
        n_plots:     Número de parcelas.
        individuals: Número de robots.
        r_r:         Radio de repulsión en búsqueda (m).
        o_r:         Radio de orientación en búsqueda (m).
        a_r:         Radio de atracción en búsqueda (m).
        r_r_deliver: Radio de repulsión en entrega (m).
        o_r_deliver: Radio de orientación en entrega (m).
        a_r_deliver: Radio de atracción en entrega (m).
        nest_ri:     Radio del sensor de nest (m).
        nest_rs:     Radio de emisión del nest (m).
        obj_r:       Radio del sensor de objetos, desde el borde del robot (m).
        time_limit:  Iteraciones máximas.

    Returns:
        Parámetros completos para run() (sin animation ni seed).
    """
    return console.prompt_parameters(
        {
            "n_objects": n_objects, "n_plots": n_plots, "individuals": individuals,
            "r_r": r_r, "o_r": o_r, "a_r": a_r,
            "r_r_deliver": r_r_deliver, "o_r_deliver": o_r_deliver, "a_r_deliver": a_r_deliver,
            "nest_ri": nest_ri, "nest_rs": nest_rs, "obj_r": obj_r,
            "time_limit": time_limit,
        },
        max_plots(),
    )


def single_run(
    n_objects:   int | None   = None,
    n_plots:     int | None   = None,
    individuals: int | None   = None,
    r_r:         float | None = None,
    o_r:         float | None = None,
    a_r:         float | None = None,
    animation:   bool | None  = None,
    r_r_deliver: float | None = None,
    o_r_deliver: float | None = None,
    a_r_deliver: float | None = None,
    nest_ri:     float | None = None,
    nest_rs:     float | None = None,
    obj_r:       float | None = None,
    time_limit:  int | None   = None,
    save_dir:    str | Path | None = config.RESULTS_DIR,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ejecuta una simulación con barra de progreso y tabla de resultados.

    Los parámetros que no se den se piden por consola. Si save_dir no es
    None, crea dentro una carpeta con fecha y hora (ver _make_run_dir) y
    guarda ahí farming_report.npy, farming_objects_report.npy, el video de la
    animación y el reporte de texto config.REPORT_FILENAME con los parámetros
    y las funciones objetivo.

    Args:
        n_objects:   Número de objetos.
        n_plots:     Número de parcelas (vacío en consola = el máximo que cabe).
        individuals: Número de robots.
        r_r:         Radio de repulsión en búsqueda (m).
        o_r:         Radio de orientación en búsqueda (m).
        a_r:         Radio de atracción en búsqueda (m).
        animation:   Reproducir o grabar la animación; se pide como YES/NO si no se da.
        r_r_deliver: Radio de repulsión en entrega (m).
        o_r_deliver: Radio de orientación en entrega (m).
        a_r_deliver: Radio de atracción en entrega (m).
        nest_ri:     Radio del sensor de nest (m).
        nest_rs:     Radio de emisión del nest (m).
        obj_r:       Radio del sensor de objetos, desde el borde del robot (m).
        time_limit:  Iteraciones máximas.
        save_dir:    Carpeta raíz de resultados; None = no guardar archivos.

    Returns:
        Tupla (report, objects_report, metrics), igual que run().
    """
    p = _collect_params(n_objects, n_plots, individuals, r_r, o_r, a_r,
                        r_r_deliver, o_r_deliver, a_r_deliver, nest_ri, nest_rs, obj_r,
                        time_limit)
    animation = console.ask_yes_no(
        "Animation? (YES/NO) [Enter = YES]: ", animation, default=config.FARMING_DEFAULT_ANIMATION
    )

    max_iter = int(p["time_limit"])
    print(f"  Max iterations: {max_iter}")

    run_dir = _make_run_dir(save_dir, "single") if save_dir is not None else None

    progress_bar = tqdm(total=max_iter, desc="Farming   ", unit="iter", ncols=72, colour="green")
    progress = console.TqdmProgress(progress_bar, p["n_objects"])

    t0 = time.time()
    report, objects_report, metrics = run(
        animation=animation, progress_callback=progress, output_dir=run_dir, **p
    )
    progress.finish(report.shape[0], round(metrics["success_ratio"] * p["n_objects"]))
    elapsed = time.time() - t0

    console.print_results(metrics, elapsed, report.shape[0], max_iter, p)

    if run_dir is not None:
        np.save(run_dir / "farming_report", report)
        np.save(run_dir / "farming_objects_report", objects_report)
        files = [_REPORT_FILE_DESCRIPTION, _OBJECTS_FILE_DESCRIPTION]
        if animation and config.SAVE_VIDEO:
            files.append(f"{config.VIDEO_FILENAME}: animation video")
        console.write_run_report(
            run_dir / config.REPORT_FILENAME, p, metrics, report.shape[0], elapsed,
            config.SEED, files,
        )
        print(f"  Results saved in: {run_dir}")

    return report, objects_report, metrics


def statistical_run(
    replicas:    int,
    n_objects:   int | None   = None,
    n_plots:     int | None   = None,
    individuals: int | None   = None,
    r_r:         float | None = None,
    o_r:         float | None = None,
    a_r:         float | None = None,
    r_r_deliver: float | None = None,
    o_r_deliver: float | None = None,
    a_r_deliver: float | None = None,
    nest_ri:     float | None = None,
    nest_rs:     float | None = None,
    obj_r:       float | None = None,
    time_limit:  int | None   = None,
    save_dir:    str | Path | None = config.RESULTS_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ejecuta varias réplicas sin animación, con barra de progreso y tabla estadística.

    La réplica r usa la semilla config.SEED + r. Los parámetros que no se
    den se piden por consola y se aplican a todas las réplicas. Si save_dir
    no es None, crea dentro una carpeta con fecha y hora (ver _make_run_dir)
    y guarda ahí farming_metrics_report.npy, farming_running_mean.npy,
    farming_final_mean.npy y el reporte de texto config.REPORT_FILENAME con
    los parámetros y las funciones objetivo de cada réplica.

    Args:
        replicas:    Número de réplicas (al menos 1).
        n_objects:   Número de objetos.
        n_plots:     Número de parcelas (vacío en consola = el máximo que cabe).
        individuals: Número de robots.
        r_r:         Radio de repulsión en búsqueda (m).
        o_r:         Radio de orientación en búsqueda (m).
        a_r:         Radio de atracción en búsqueda (m).
        r_r_deliver: Radio de repulsión en entrega (m).
        o_r_deliver: Radio de orientación en entrega (m).
        a_r_deliver: Radio de atracción en entrega (m).
        nest_ri:     Radio del sensor de nest (m).
        nest_rs:     Radio de emisión del nest (m).
        obj_r:       Radio del sensor de objetos, desde el borde del robot (m).
        time_limit:  Iteraciones máximas.
        save_dir:    Carpeta raíz de resultados; None = no guardar archivos.

    Returns:
        Tupla (metrics_report, running_mean, final_mean): funciones objetivo
        [f1, f2, f3, f4] por réplica, shape (R, 4); su media acumulada tras
        cada réplica, shape (R, 4); y la media de todas, shape (4,).

    Raises:
        ValueError: Si replicas < 1.
    """
    if replicas < 1:
        raise ValueError(f"replicas={replicas} debe ser al menos 1.")
    p = _collect_params(n_objects, n_plots, individuals, r_r, o_r, a_r,
                        r_r_deliver, o_r_deliver, a_r_deliver, nest_ri, nest_rs, obj_r,
                        time_limit)
    max_iter = int(p["time_limit"])

    metrics_report = np.zeros((replicas, 4))
    running_mean   = np.zeros((replicas, 4))

    rep_bar = tqdm(total=replicas, desc="Replicas  ", unit="rep", ncols=72, colour="cyan")
    rep_bar.set_postfix(obj=f"0/{p['n_objects']}")

    t0 = time.time()
    for r in range(replicas):
        iter_bar = tqdm(
            total=max_iter, desc=f"  Rep {r + 1:>3}/{replicas}", unit="iter",
            ncols=72, leave=False, colour="green",
        )
        progress = console.TqdmProgress(iter_bar, p["n_objects"])
        _, _, metrics = run(animation=False, seed=config.SEED + r, progress_callback=progress, **p)
        iter_bar.close()

        metrics_report[r] = [
            metrics["load_balance"], metrics["mean_energy"], metrics["success_ratio"],
            metrics["signaling_cost"],
        ]
        delivered = round(metrics["success_ratio"] * p["n_objects"])
        rep_bar.set_postfix(obj=f"{delivered}/{p['n_objects']}")
        rep_bar.update(1)

    rep_bar.close()
    elapsed = time.time() - t0

    final_mean = np.mean(metrics_report, axis=0)
    for r in range(replicas):
        running_mean[r] = np.mean(metrics_report[:r + 1], axis=0)

    console.print_stats(metrics_report, elapsed, replicas)

    if save_dir is not None:
        run_dir = _make_run_dir(save_dir, "stats")
        np.save(run_dir / "farming_metrics_report", metrics_report)
        np.save(run_dir / "farming_running_mean", running_mean)
        np.save(run_dir / "farming_final_mean", final_mean)
        console.write_stats_report(
            run_dir / config.REPORT_FILENAME, p, metrics_report, elapsed,
            [config.SEED + r for r in range(replicas)],
            [
                "farming_metrics_report.npy: [f1, f2, f3, f4] per replica, shape (R, 4)",
                "farming_running_mean.npy: running mean after each replica, shape (R, 4)",
                "farming_final_mean.npy: mean over all replicas, shape (4,)",
            ],
        )
        print(f"  Results saved in: {run_dir}")

    return metrics_report, running_mean, final_mean
