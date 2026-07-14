# -*- coding: utf-8 -*-
"""
Simulacion de tarea de farming (recoleccion agricola) con el modelo RAOI.

En la tarea de farming el enjambre debe recolectar objetos distribuidos en
franjas a ambos lados de parcelas agricolas (segmentos horizontales dentro
del area) y depositarlos en el nest. No existe un objectbox fijo: los objetos
estan repartidos en el escenario segun la geometria de las parcelas.

Parcelas:
  Segmentos horizontales de longitud fija (FARMING_PLOT_LENGTH), distribuidos
  en pares simetricos respecto al centro del area con separacion constante
  (FARMING_PLOT_SEPARATION centro a centro). El numero maximo de parcelas que
  caben se calcula automaticamente y se valida al inicio de run().

Line-of-sight:
  Un robot no puede detectar un objeto si alguna parcela intersecta
  fisicamente el segmento robot -> objeto. La verificacion usa interseccion
  de segmentos 2D simplificada (O(n_plots) por robot por iteracion).

Loop de simulacion:
  Termina al completar la tarea (todos los objetos entregados) o al agotar max_iter.
  Al finalizar reporta objetos entregados y los que quedaron pendientes.

Obstaculos:
  Las parcelas actuan como barreras de repulsion virtual — los robots las
  esquivan con la misma mecanica que las paredes del area.

Lógica de influencia:
  - Sin objeto: influencia hacia el objeto detectado disponible más cercano.
  - Con objeto: influencia hacia el nest, dentro de su radio de detección.

Referencia:
  Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
  based on repulsion, attraction and influence. Autonomous Robots.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autonoma de Nuevo Leon
"""

import math
import random
import time
from typing import Optional, Callable

import numpy as np
from tqdm import tqdm

from . import config
from . import metrics as mtr
from . import visualization as viz
from .behavior import (
    wrap_angle,
    repulsion_vector,
    combined_direction,
    detect_neighbors,
    select_voltage,
)
from .dynamics import DynamicsConstants, integrate_robot
from .environment import detect_walls, detect_plots


# ============================================================================
# Utilidades internas
# ============================================================================

def _compute_max_plots(
    area_limits:    float,
    separation:     float,
    strip_width:    float,
    plot_repulsion: float,
    margin_y:       float,
) -> int:
    """
    Calcula el numero maximo de parcelas que caben en el area.

    El bloque de n parcelas es (n-1)*separation + 2*(strip_width+plot_repulsion)
    y debe ser <= (area_limits - 2*margin_y).

    Args:
        area_limits:    Lado del area cuadrada (m).
        separation:     Distancia centro a centro entre parcelas (m).
        strip_width:    Ancho de franja de objetos a cada lado (m).
        plot_repulsion: Radio de repulsion de parcelas (m).
        margin_y:       Margen en Y en cada extremo (m).

    Returns:
        Numero maximo de parcelas.
    """
    disponible = area_limits - 2.0 * margin_y
    n = 1
    while True:
        bloque = (n - 1) * separation + 2.0 * (strip_width + plot_repulsion)
        if bloque > disponible:
            return max(1, n - 1)
        n += 1


def _build_plots_symmetric(
    n_plots:      int,
    plot_length:  float,
    area_limits:  float,
    separation:   float,
) -> list:
    """
    Genera segmentos de parcela simetricos desde el centro del area.

    Con n_plots par: pares simetricos alrededor de y=area/2.
    Con n_plots impar: parcela central + pares hacia afuera.
    El segmento se centra en X dejando pasillos laterales iguales.

    Args:
        n_plots:      Numero de parcelas.
        plot_length:  Longitud de cada segmento (m).
        area_limits:  Lado del area cuadrada (m).
        separation:   Distancia centro a centro entre parcelas (m).

    Returns:
        Lista de dicts {'x0', 'x1', 'y'} ordenados de menor a mayor y.
    """
    center_y = area_limits / 2.0
    x0 = (area_limits - plot_length) / 2.0
    x1 = x0 + plot_length

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


def _segment_blocked_by_plots(
    robot_pos:  np.ndarray,
    target_pos: np.ndarray,
    plots:      list,
) -> bool:
    """
    Verifica si alguna parcela intersecta la linea robot -> objetivo.

    Dado que las parcelas son horizontales (y constante), la interseccion
    se resuelve analiticamente en O(1) por parcela:
      t = (y_plot - robot.y) / (obj.y - robot.y)
      x_cross = robot.x + t * (obj.x - robot.x)
      Bloqueado si t in (0,1) y x_cross in [x0, x1].

    Args:
        robot_pos:  Posicion [x, y] del robot (m).
        target_pos: Posicion [x, y] del objetivo (m).
        plots:      Lista de dicts {'x0', 'x1', 'y'}.

    Returns:
        True si alguna parcela bloquea la linea de vision.
    """
    rx, ry = float(robot_pos[0]), float(robot_pos[1])
    tx, ty = float(target_pos[0]), float(target_pos[1])
    dy     = ty - ry

    if abs(dy) < 1e-9:
        return False   # misma Y, ningun segmento horizontal bloquea

    for plot in plots:
        yp = float(plot["y"])
        if (ry < yp < ty) or (ty < yp < ry):
            t       = (yp - ry) / dy
            x_cross = rx + t * (tx - rx)
            if float(plot["x0"]) <= x_cross <= float(plot["x1"]):
                return True

    return False


def _resolve_plot_collision(
    c_old: np.ndarray,
    c_new: np.ndarray,
    plots: list,
) -> tuple[np.ndarray, bool]:
    """
    Impide que el robot atraviese fisicamente una parcela.

    Verifica si el desplazamiento de este paso (c_old -> c_new) cruza el
    segmento horizontal de alguna parcela, con la misma matematica analitica
    que _segment_blocked_by_plots (aqui aplicada al movimiento real, no a la
    linea de vision). Si hay cruce, el movimiento se cancela por completo:
    el robot vuelve a su posicion anterior y su orientacion se refleja,
    sesgada hacia continuar a lo largo de la parcela (eje X) en vez de una
    reflexion pura, para no quedar rebotando en zigzag entre dos filas
    paralelas cuando el pasillo entre ellas es angosto.

    Esto es un respaldo geometrico duro sobre la repulsion existente
    (FARMING_PLOT_REPULSION ya debe frenar al robot antes de llegar aqui en
    circunstancias normales); sirve para casos limite de angulo rasante o
    pasos de integracion grandes donde la repulsion sola no alcanza a
    prevenir el cruce.

    Args:
        c_old: Estado del robot antes de integrar este paso, shape (6,).
        c_new: Estado propuesto tras integrar (ya incluye rebote de paredes).
        plots: Lista de dicts {'x0', 'x1', 'y'}.

    Returns:
        c_fixed : c_new sin cambios si no hubo cruce; estado revertido y
                  reflejado si lo hubo.
        blocked : True si se bloqueo el cruce.
    """
    rx, ry = float(c_old[0]), float(c_old[1])
    tx, ty = float(c_new[0]), float(c_new[1])
    dy     = ty - ry

    if abs(dy) < 1e-9:
        return c_new, False

    for plot in plots:
        yp = float(plot["y"])
        if (ry < yp < ty) or (ty < yp < ry):
            t       = (yp - ry) / dy
            x_cross = rx + t * (tx - rx)
            if float(plot["x0"]) <= x_cross <= float(plot["x1"]):
                c_fixed = c_old.copy()

                # Reflexion especular (componente Y invertida) sesgada hacia
                # continuar a lo largo del eje X — la parcela es un obstaculo
                # alargado horizontal, y una reflexion pura puede rebotar al
                # robot de vuelta hacia la fila paralela opuesta si el pasillo
                # entre dos parcelas es angosto, generando un zigzag lento en
                # vez de una salida directa por el pasillo.
                mirrored = wrap_angle(-c_old[3])
                tangent  = 0.0 if math.cos(mirrored) >= 0.0 else math.pi
                bias     = 0.3
                new_theta = math.atan2(
                    (1.0 - bias) * math.sin(mirrored) + bias * math.sin(tangent),
                    (1.0 - bias) * math.cos(mirrored) + bias * math.cos(tangent),
                )
                c_fixed[3]  = wrap_angle(new_theta + np.random.normal(0.0, 0.15))
                c_fixed[4]  = 0.0
                c_fixed[5]  = 0.0
                return c_fixed, True

    return c_new, False


def _detect_influence_farming(
    robot_pos:     np.ndarray,
    robot_theta:   float,
    target_pos:    np.ndarray,
    target_radius: float,
    fov_influence: float,
    n_repulsion:   int,
    n_walls:       int,
    plots:         list,
) -> tuple[float, float, int]:
    """
    Detecta si el robot percibe un objetivo (objeto o nest) — sistema de vision.

    La influencia se suprime si hay vecinos robots en zona de repulsion,
    paredes detectadas, o si alguna parcela se interpone en la linea de
    vision robot -> objetivo (oclusion fisica, igual que un obstaculo real
    tapando la vista de una camara/sensor). Las parcelas NO producen
    repulsion sobre la senal I por si mismas (esa logica vive en el vector
    R); aqui solo actuan como bloqueadoras de vision cuando estan
    literalmente entre el robot y el objetivo.

    La distancia y el angulo percibidos incluyen ruido gaussiano
    (config.FARMING_SENSOR_DISTANCE_NOISE_STD / FARMING_SENSOR_ANGLE_NOISE_STD),
    modelando la imprecision de un sensor real: el borde de deteccion queda
    "difuso" en vez de un corte perfecto, y la direccion percibida no es
    exacta. La oclusion de vision usa las posiciones reales (es un hecho
    fisico, no una medicion con error).

    Args:
        robot_pos:     Posicion [x, y] del robot (m).
        robot_theta:   Orientacion del robot (rad).
        target_pos:    Posicion [x, y] del objetivo.
        target_radius: Radio de deteccion del objetivo (m).
        fov_influence: Campo de vision de influencia (rad).
        n_repulsion:   Numero de vecinos robots en zona R (suprime influencia).
        n_walls:       Paredes detectadas (suprime influencia).
        plots:         Lista de dicts {'x0','x1','y'} — parcelas que pueden
                       ocluir la vision hacia el objetivo.

    Returns:
        distance : Distancia percibida al objetivo (m), con ruido de sensor.
        angle    : Angulo percibido hacia el objetivo (rad), con ruido de sensor.
        detected : 1 si detectado, 0 si no.
    """
    dx            = float(target_pos[0]) - robot_pos[0]
    dy            = float(target_pos[1]) - robot_pos[1]
    true_distance = math.sqrt(dx ** 2 + dy ** 2)
    true_angle    = wrap_angle(math.atan2(dy, dx))

    distance = max(0.0, true_distance + np.random.normal(0.0, config.FARMING_SENSOR_DISTANCE_NOISE_STD))
    angle    = wrap_angle(true_angle + np.random.normal(0.0, config.FARMING_SENSOR_ANGLE_NOISE_STD))

    beta   = wrap_angle(angle - robot_theta)
    gamma  = wrap_angle(robot_theta - angle)
    i_diff = min(beta, gamma)

    detected = 0
    if (target_radius > 0.0   # radio 0 significa "sin objetivo": nunca detecta
            and i_diff < fov_influence / 2
            and distance <= target_radius
            and n_repulsion == 0 and n_walls == 0
            and not _segment_blocked_by_plots(robot_pos, target_pos, plots)):
        detected = 1

    return distance, angle, detected




def _spawn_objects_farming(
    n_objects:      int,
    plots:          list,
    strip_width:    float,
    plot_repulsion: float,
    area_limits:    float,
    rng:            np.random.Generator,
) -> np.ndarray:
    """
    Distribuye objetos uniformemente entre parcelas en franjas a ambos lados.

    Cada parcela recibe floor(n_objects/n_plots) objetos, con los restantes
    en las primeras parcelas. Dentro de cada parcela, mitad arriba y mitad
    abajo del segmento.

    Args:
        n_objects:      Numero total de objetos.
        plots:          Lista de dicts {'x0', 'x1', 'y'}.
        strip_width:    Ancho de la franja a cada lado (m).
        plot_repulsion: Radio de repulsion (define el margen minimo) (m).
        area_limits:    Lado del area cuadrada (m).
        rng:            Generador NumPy para reproducibilidad.

    Returns:
        Posiciones de objetos, shape (n_objects, 2).
    """
    n_plots = len(plots)
    margin  = plot_repulsion + 0.1
    base    = n_objects // n_plots
    extra   = n_objects  % n_plots

    positions = []
    for p_idx, plot in enumerate(plots):
        n_here  = base + (1 if p_idx < extra else 0)
        x0, x1  = plot["x0"], plot["x1"]
        yp      = plot["y"]

        y_lo_up = yp + margin
        y_hi_up = min(yp + margin + strip_width, area_limits - margin)
        y_lo_dn = max(yp - margin - strip_width, margin)
        y_hi_dn = yp - margin

        n_up = n_here // 2
        n_dn = n_here - n_up

        for _ in range(n_up):
            positions.append([
                rng.uniform(x0, x1),
                rng.uniform(y_lo_up, max(y_lo_up + 0.1, y_hi_up)),
            ])
        for _ in range(n_dn):
            positions.append([
                rng.uniform(x0, x1),
                rng.uniform(max(y_lo_dn, margin), y_hi_dn),
            ])

    return np.array(positions, dtype=float)


def _free_zones_y(
    plots:          list,
    plot_repulsion: float,
    area_limits:    float,
) -> list:
    """
    Calcula intervalos de Y libres de parcelas para el spawn de robots.

    Solo respeta el radio de repulsion de la parcela — no la franja de objetos.

    Args:
        plots:          Lista de dicts de parcelas.
        plot_repulsion: Radio de repulsion de parcelas (m).
        area_limits:    Lado del area (m).

    Returns:
        Lista de (y_lo, y_hi) con intervalos libres en Y.
    """
    blocked = []
    margin  = plot_repulsion + 0.1
    for plot in plots:
        yp = plot["y"]
        blocked.append((yp - margin, yp + margin))

    blocked.sort(key=lambda b: b[0])
    free   = []
    y_prev = 0.0
    for lo, hi in blocked:
        if lo > y_prev + 1.0:
            free.append((y_prev, lo))
        y_prev = hi
    if y_prev < area_limits - 1.0:
        free.append((y_prev, area_limits))

    return free


# ============================================================================
# Simulacion principal
# ============================================================================

def run(
    n_objects:   int,
    n_plots:     int,
    individuals: int,
    r_r:         float,
    o_r:         float,
    a_r:         float,
    animation:   bool,
    seed:        Optional[int] = None,
    nest_ri:     Optional[float] = None,
    nest_rs:     Optional[float] = None,
    obj_r:       Optional[float] = None,
    time_limit:  Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Ejecuta la simulación de la tarea de farming con el modelo RAOI.

    El bucle termina al completar la tarea (todos los objetos entregados) o
    al agotar max_iter, lo que ocurra primero. Al finalizar reporta objetos
    entregados y los que quedaron pendientes.

    El robot se simula a escala física (ver config.FARMING_ROBOT_SCALE):
    dimensiones lineales × escala, masa × escala³, inercia × escala⁵ —
    escalado como sólido uniforme.

    Args:
        n_objects:   Número de objetos a recolectar.
        n_plots:     Número de parcelas (surcos) en el escenario.
        individuals: Número de robots en el enjambre.
        r_r:  Radio adicional de repulsión   (m, sumado al cuerpo del robot escalado).
        o_r:  Radio adicional de orientación (m, sumado al cuerpo del robot escalado).
        a_r:  Radio adicional de atracción   (m, sumado al cuerpo del robot escalado).
        animation: Si True, reproduce la animación Pygame al terminar.
        seed: Semilla aleatoria. None -> usa config.SEED.
        nest_ri: Radio sensor del robot hacia el nest (m). None -> config.FARMING_NEST_RI.
        nest_rs: Radio de emisión del nest (m), se suma a nest_ri para el
                 rango efectivo de detección. None -> config.FARMING_NEST_RS.
                 El nest emite una señal activa (modelo ri + rs).
        obj_r:   Radio de detección del robot hacia un objeto (m), sensor
                 tipo cámara — el objeto no emite señal propia, es un único
                 componente sin rs. None -> config.FARMING_OBJECT_DETECTION_RADIUS.
        time_limit: Si se especifica, reemplaza config.FARMING_DEFAULT_MAX_ITER:
                 la simulación corre exactamente time_limit iteraciones (o
                 menos si se entregan todos los objetos antes). Útil para
                 escenarios con muchos objetos donde se quiere medir la
                 fracción entregada dentro de una ventana de tiempo fija,
                 en vez de dejar que la tarea determine su propia duración.
                 None -> usa config.FARMING_DEFAULT_MAX_ITER.
        progress_callback: f(t, max_iter) llamada al final de cada iteración.

    Returns:
        report         : Estado del enjambre, shape (T, N, 8).
        objects_report : Posiciones de objetos, shape (T, O, 2).
        metrics        : Dict de métricas (ver metrics.farming_metrics).
    """
    _seed = seed if seed is not None else config.SEED
    random.seed(_seed)
    np.random.seed(_seed)
    rng = np.random.default_rng(_seed)

    # ── Robot físico escalado ──────────────────────────────────────────────────
    scale        = config.FARMING_ROBOT_SCALE
    body_radius  = config.ROBOT_BODY_RADIUS * scale
    farming_v_max = config.V_MAX_LINEAR * scale
    dyn = DynamicsConstants(
        mass      = config.ROBOT_MASS      * scale ** 3,
        inertia   = config.ROBOT_INERTIA   * scale ** 5,
        d         = config.ROBOT_D         * scale,
        wheel_r   = config.ROBOT_WHEEL_R   * scale,
        wheel_sep = config.ROBOT_WHEEL_SEP * scale,
        # Motor (Ts/Ks/Kl) NO se escala — sin datos reales del motor del
        # robot grande, escalarlo seria inventar numeros. OMEGA_MAX tampoco
        # (ver docstring de config.FARMING_ROBOT_SCALE); V_MAX_LINEAR si se
        # escala (farming_v_max), pasado como override a integrate_robot().
    )

    # ── Parametros ────────────────────────────────────────────────────────────
    area_limits    = config.FARMING_AREA_LIMITS
    weights        = config.FARMING_RAOI_WEIGHTS
    fov            = config.RAOI_FOV
    voltages       = config.VOLTAGE

    r_repulsion    = body_radius + r_r
    r_orientation  = body_radius + o_r
    r_attraction   = body_radius + a_r

    plot_length    = config.FARMING_PLOT_LENGTH
    plot_repulsion = config.FARMING_PLOT_REPULSION
    strip_width    = config.FARMING_STRIP_WIDTH
    separation     = config.FARMING_PLOT_SEPARATION
    margin_y       = config.FARMING_PLOT_MARGIN_Y
    nest_pos       = np.array(config.FARMING_NEST_POSITION, dtype=float)
    nest_ri        = nest_ri if nest_ri is not None else config.FARMING_NEST_RI
    nest_rs        = nest_rs if nest_rs is not None else config.FARMING_NEST_RS
    nest_area      = config.FARMING_NEST_AREA_SIDE
    pick_radius    = config.FARMING_PICK_RADIUS
    deposit_radius = config.FARMING_DEPOSIT_RADIUS
    obj_r          = obj_r if obj_r is not None else config.FARMING_OBJECT_DETECTION_RADIUS

    # Validar que la parcela quepa en X dentro del area (con margen minimo
    # a cada lado para el corredor de spawn). Antes esto no se validaba y
    # una parcela mas larga que el area quedaba con extremos fuera del
    # area sin ningun aviso.
    if plot_length >= area_limits:
        raise ValueError(
            f"FARMING_PLOT_LENGTH={plot_length} m no cabe en un area de "
            f"{area_limits}x{area_limits} m: la parcela quedaria mas larga "
            f"que el area misma. Reduce FARMING_PLOT_LENGTH o aumenta "
            f"FARMING_AREA_LIMITS en config.py."
        )

    # Validar n_plots
    max_plots = _compute_max_plots(
        area_limits, separation, strip_width, plot_repulsion, margin_y
    )
    if n_plots > max_plots:
        raise ValueError(
            f"n_plots={n_plots} excede el maximo permitido ({max_plots}) "
            f"para area {area_limits}x{area_limits} m con los parametros actuales."
        )

    # max_iter: tiempo limite para completar la tarea.
    # El loop termina antes si todos los objetos son entregados (nest_remaining==0).
    # time_limit, si se especifica, reemplaza este valor (modo f3: medir la
    # fraccion entregada dentro de una ventana de tiempo fija, tipicamente
    # con muchos objetos donde no se espera terminar antes).
    max_iter = int(time_limit) if time_limit is not None else config.FARMING_DEFAULT_MAX_ITER

    # ── Geometria ─────────────────────────────────────────────────────────────
    plots = _build_plots_symmetric(n_plots, plot_length, area_limits, separation)

    # ── Arrays de estado ──────────────────────────────────────────────────────
    C                   = np.zeros((individuals, 6))
    report_buf          = np.zeros((max_iter, individuals, 8))
    obj_report_buf      = np.zeros((max_iter, n_objects, 2))
    carrying_report_buf = np.full((max_iter, individuals), -1, dtype=int)
    state_detected      = np.zeros((max_iter, individuals))
    free_iters          = np.zeros(individuals, dtype=int)
    state_prev          = np.full(individuals, -1, dtype=int)

    # ── Estado de objetos ─────────────────────────────────────────────────────
    objects_pos    = _spawn_objects_farming(
        n_objects, plots, strip_width, plot_repulsion, area_limits, rng
    )
    obv            = np.ones(n_objects, dtype=int)
    delivered      = np.zeros(n_objects, dtype=bool)
    grip_state     = np.zeros(individuals, dtype=int)
    nest_influence = np.zeros(individuals, dtype=int)
    carrying       = np.full(individuals, -1, dtype=int)
    target_object  = np.full(individuals, -1, dtype=int)
    levy_run_remaining = np.zeros(individuals, dtype=int)
    edge_follow_side   = np.zeros(individuals, dtype=int)  # 0=indeciso, +1/-1=lado elegido

    # Metricas auxiliares: [delivery_time, search_time, n_collected, total_iters]
    collected_objects = np.zeros((individuals, 4))

    # ── Spawn de robots en linea vertical (fuera del nest) ────────────────────
    # Columna vertical en x=FARMING_SPAWN_LINE_X, robots equiespaciados en Y,
    # todos con la misma orientacion inicial (hacia las parcelas). Evita el
    # amontonamiento dentro del area cuadrada del nest cuando individuals es
    # grande, y reduce el desplazamiento inicial hasta la primera deteccion util.
    #
    # spawn_x: al menos FARMING_SPAWN_LINE_X, pero nunca dentro de la zona de
    # repulsion de la pared oeste (r_repulsion). Si quedara dentro, la logica
    # de rebote de pared en integrate_robot se dispara desde la primera
    # iteracion y puede invertir la orientacion inicial del robot.
    spawn_x = max(config.FARMING_SPAWN_LINE_X, r_repulsion + 0.1)

    # Espaciado adaptativo: se intenta el espaciado "ideal" (mas aire entre
    # robots, ver FARMING_SPAWN_SPACING_FACTOR), pero si no caben todos en
    # el area con ese espaciado, se reduce hasta el minimo seguro (2*r_repulsion,
    # el punto donde las zonas de repulsion mutua apenas no se tocan) antes
    # de rendirse. Solo se lanza error si ni siquiera el minimo seguro cabe —
    # es decir, si de verdad no hay espacio fisico para todos los robots.
    safety_spacing = max(config.SPAWN_MIN_SEPARATION, 2.0 * r_repulsion)
    ideal_spacing  = max(config.SPAWN_MIN_SEPARATION,
                         config.FARMING_SPAWN_SPACING_FACTOR * r_repulsion)

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
        print(
            f"  Nota: con individuals={individuals}, el espaciado de spawn se "
            f"redujo de {ideal_spacing:.3f} m (ideal) a {spawn_spacing:.3f} m "
            f"para que todos quepan en el area (sigue siendo seguro: "
            f">= {safety_spacing:.3f} m)."
        )

    line_span = (individuals - 1) * spawn_spacing
    y_start   = max(spawn_spacing, (area_limits - line_span) / 2.0)

    for i in range(individuals):
        C[i, 0] = spawn_x
        C[i, 1] = y_start + i * spawn_spacing
        C[i, 2] = 0.0
        C[i, 3] = 0.0   # orientacion uniforme hacia +X (hacia las parcelas)
        C[i, 4] = 0.0
        C[i, 5] = 0.0

    dir_explore = C[:, 3].copy()

    # ── Loop principal — termina al completar tarea o agotar max_iter ───────────
    t = 0
    nest_remaining  = n_objects
    while t < max_iter and nest_remaining > 0:
        desired_voltages = np.zeros((individuals, 2))
        desired_thetas   = np.zeros(individuals)

        for i in range(individuals):
            noise        = np.random.normal(0.0, config.FARMING_ACTUATOR_NOISE_STD)
            theta_before = C[i, 3]

            walls    = detect_walls(C[i, :2], r_repulsion, area_limits)
            plot_pts = detect_plots(C[i, :2], plots, plot_repulsion)
            if not plot_pts:
                edge_follow_side[i] = 0  # fuera de contacto: reiniciar para el proximo encuentro
            neighbors = detect_neighbors(
                i, C, r_repulsion, r_orientation, r_attraction,
                fov, len(walls) + len(plot_pts),
            )

            # ── Objetivo de influencia ──────────────────────────────────────────
            #
            # Fase 1 (sin objeto, nest_influence==0):
            #   En cada iteracion se recalcula cual es el objeto detectado
            #   (dentro de obj_r, con linea de vision libre) mas cercano al
            #   robot en este instante. No se guarda un objetivo persistente
            #   entre iteraciones: el robot solo reacciona a lo que su sensor
            #   le muestra ahora mismo, igual que un robot real — si algo se
            #   le cruza enfrente lo nota y puede cambiar de objetivo de
            #   inmediato, sin necesidad de ningun mecanismo de timeout.
            #
            # Fase 2 (con objeto, nest_influence==1):
            #   Senal directa hacia el nest. La composicion R + I en
            #   combined_direction rodea las parcelas de forma emergente,
            #   sin necesitar un punto de entrada intermedio.
            #
            # Radios realistas (obj_r ~1 m, nest_ri+nest_rs ~5 m) + sistema de
            # vision: si una parcela se interpone en la linea robot->objetivo,
            # no hay deteccion (_detect_influence_farming), sin importar
            # distancia o FOV. Con radios tan cortos la busqueda depende
            # fuertemente de la exploracion libre y del barrido colectivo del
            # enjambre — es la contrapartida esperada de un sensor de corto
            # alcance realista frente a un beacon de area.
            #
            # La repulsion de parcelas (vector R) es independiente de esto:
            # sigue activa siempre que el robot este cerca de una parcela,
            # sea que la perciba (I) o no.

            if nest_influence[i] == 0:
                available = np.where(obv == 1)[0]
                best_idx  = -1
                best_dist = math.inf
                for o in available:
                    d = float(np.linalg.norm(C[i, :2] - objects_pos[o]))
                    if (d <= obj_r and d < best_dist
                            and not _segment_blocked_by_plots(
                                C[i, :2], objects_pos[o], plots)):
                        best_dist = d
                        best_idx  = int(o)
                target_object[i] = best_idx   # -1 si no se detecta nada ahora

                if target_object[i] >= 0:
                    target_pos    = objects_pos[target_object[i]]
                    target_radius = obj_r
                else:
                    # Nada detectado en este instante — explorar libremente
                    target_pos    = C[i, :2]
                    target_radius = 0.0

            else:
                # Fase 2: con objeto — navegar al nest.
                # Las parcelas entre el robot y el nest producen repulsion R.
                # La composicion R + I de combined_direction hace que el robot
                # las rodee de forma emergente: R lo aleja de la parcela e I
                # lo jala hacia el nest, la resultante es una trayectoria en
                # arco alrededor del extremo de la parcela.
                # No se necesitan waypoints — esto es comportamiento RAOI puro.
                # El nest SI emite señal activa: rango efectivo = ri + rs.
                target_pos    = nest_pos
                target_radius = nest_ri + nest_rs

            # ── Deteccion de influencia ───────────────────────────────────────
            # FOV 360 en ambas fases: el objetivo es una senal de area (mismo
            # criterio que ya se usaba para el nest), no requiere que el
            # robot este mirando de frente para percibirla.
            fov_detect = 2 * math.pi
            inf_dist, inf_angle, inf_detected = _detect_influence_farming(
                C[i, :2], C[i, 3],
                target_pos, target_radius,
                fov_detect,
                neighbors["n_rep"], len(walls),
                plots,
            )

            # ── Vectores RAOI ─────────────────────────────────────────────────
            active = {}

            all_repulsion = neighbors["rep_neighbors"] + walls + plot_pts
            if all_repulsion:
                rvx, rvy = repulsion_vector(C[i, :2], all_repulsion)
                r_norm   = max(math.sqrt(rvx ** 2 + rvy ** 2), 1e-9)
                rvx, rvy = rvx / r_norm, rvy / r_norm

                if plot_pts:
                    # Bordeo: la parcela es un obstaculo alargado (75 m), no
                    # puntual — mezclar con una componente tangencial (90°
                    # rotada) para favorecer recorrer el borde en vez de
                    # rebotar y quedar oscilando cerca del mismo punto.
                    #
                    # El lado (sentido de giro) se decide una sola vez al
                    # entrar en contacto con la parcela y se mantiene fijo
                    # mientras el robot siga repelido por ella (histeresis).
                    # Sin esto, recalcular el lado "mas alineado con el
                    # heading actual" en cada iteracion crea un ciclo de
                    # retroalimentacion: la fisica (inercia + control
                    # proporcional) no alcanza el angulo deseado tan rapido
                    # como cambia la eleccion, y el robot queda oscilando
                    # entre las dos tangentes sin avanzar.
                    tx1, ty1 = -rvy, rvx
                    tx2, ty2 =  rvy, -rvx

                    if edge_follow_side[i] == 0:
                        curr_x, curr_y = math.cos(C[i, 3]), math.sin(C[i, 3])
                        d1 = tx1 * curr_x + ty1 * curr_y
                        d2 = tx2 * curr_x + ty2 * curr_y
                        edge_follow_side[i] = 1 if d1 >= d2 else -1

                    if edge_follow_side[i] == 1:
                        tx, ty = tx1, ty1
                    else:
                        tx, ty = tx2, ty2

                    ew = config.FARMING_EDGE_FOLLOW_WEIGHT
                    bx = (1.0 - ew) * rvx + ew * tx
                    by = (1.0 - ew) * rvy + ew * ty
                    b_n = max(math.sqrt(bx ** 2 + by ** 2), 1e-9)
                    rvx, rvy = bx / b_n, by / b_n

                active["R"] = (rvx, rvy)

            if neighbors["n_ori"] > 0:
                ovx = sum(neighbors["ox"]); ovy = sum(neighbors["oy"])
                o_n = max(math.sqrt(ovx ** 2 + ovy ** 2), 1e-9)
                active["O"] = (ovx / o_n, ovy / o_n)

            if neighbors["n_att"] > 0:
                avx = sum(neighbors["ax"]); avy = sum(neighbors["ay"])
                a_n = max(math.sqrt(avx ** 2 + avy ** 2), 1e-9)
                active["A"] = (avx / a_n, avy / a_n)

            if inf_detected:
                active["I"] = (math.cos(inf_angle), math.sin(inf_angle))

            if   active.get("R"): state_now = 1
            elif active.get("I"): state_now = 4
            elif active.get("O"): state_now = 3
            elif active.get("A"): state_now = 2
            else:                 state_now = 0

            state_detected[t, i] = state_now

            if active:
                desired_thetas[i] = combined_direction(C[i, 3], active, weights)
            else:
                if free_iters[i] < config.EXPLORE_FREE_ITERS:
                    free_iters[i]    += 1
                    desired_thetas[i] = C[i, 3]
                elif levy_run_remaining[i] <= 0:
                    # Fin de la racha recta: nueva racha con duracion de cola
                    # pesada (Pareto) y giro grande hacia una direccion nueva.
                    run_len = int(np.clip(
                        (rng.pareto(config.FARMING_LEVY_EXPONENT) + 1.0)
                        * config.FARMING_LEVY_MIN_RUN,
                        config.FARMING_LEVY_MIN_RUN,
                        config.FARMING_LEVY_MAX_RUN,
                    ))
                    levy_run_remaining[i] = run_len
                    dir_explore[i]        = wrap_angle(rng.uniform(0.0, 2 * math.pi))
                    desired_thetas[i]     = dir_explore[i]
                else:
                    # Dentro de la racha: seguir recto, con solo un ruido
                    # pequeno (no un giro gaussiano completo) para no ser
                    # una linea perfectamente rigida.
                    levy_run_remaining[i] -= 1
                    dir_explore[i]         = wrap_angle(
                        dir_explore[i] + np.random.normal(0.0, config.EXPLORE_TURN_NOISE * 0.2)
                    )
                    desired_thetas[i] = dir_explore[i]

            if active and state_prev[i] == 0:
                free_iters[i]         = 0
                levy_run_remaining[i] = 0
                dir_explore[i] = wrap_angle(
                    C[i, 3] + np.random.normal(0.0, config.DIREXP_RESET_NOISE)
                )

            if active:
                xT = 0.5 * math.cos(C[i, 3]) + 0.5 * math.cos(desired_thetas[i])
                yT = 0.5 * math.sin(C[i, 3]) + 0.5 * math.sin(desired_thetas[i])
                C[i, 3] = wrap_angle(math.atan2(yT, xT))
            else:
                xT = 0.5 * math.cos(C[i, 3]) + 0.5 * math.cos(dir_explore[i])
                yT = 0.5 * math.sin(C[i, 3]) + 0.5 * math.sin(dir_explore[i])
                C[i, 3] = wrap_angle(math.atan2(yT, xT))
                desired_thetas[i] = dir_explore[i]

            state_prev[i] = state_now

            desired_voltages[i] = select_voltage(
                active,
                desired_thetas[i], theta_before,
                inf_dist, target_radius, noise, voltages,
            )

        # ── Integracion dinamica ──────────────────────────────────────────────
        for i in range(individuals):
            c_old = C[i].copy()
            c_new, bounced = integrate_robot(
                c_old, desired_voltages[i], dyn, r_repulsion, area_limits,
                v_max_linear=farming_v_max,
            )
            c_new, plot_blocked = _resolve_plot_collision(c_old, c_new, plots)
            C[i] = c_new
            if bounced or plot_blocked:
                dir_explore[i] = C[i, 3]

        # ── Recoleccion y deposito ────────────────────────────────────────────
        for i in range(individuals):
            if grip_state[i] == 1 and carrying[i] >= 0:
                objects_pos[carrying[i]] = C[i, :2].copy()

            if grip_state[i] == 0:
                for o in range(n_objects):
                    if obv[o] == 0:
                        continue
                    if np.linalg.norm(C[i, :2] - objects_pos[o]) <= pick_radius:
                        grip_state[i]          = 1
                        nest_influence[i]      = 1
                        carrying[i]            = o
                        target_object[i]       = -1
                        obv[o]                 = 0
                        collected_objects[i, 2] += 1
                        break

            elif grip_state[i] == 1 and carrying[i] >= 0:
                if np.linalg.norm(C[i, :2] - nest_pos) <= deposit_radius:
                    o = carrying[i]
                    objects_pos[o]         = C[i, :2].copy()
                    delivered[o]           = True
                    obv[o]                 = 0
                    grip_state[i]          = 0
                    nest_influence[i]      = 0
                    carrying[i]            = -1
                    nest_remaining        -= 1

            if grip_state[i] == 1:
                collected_objects[i, 0] += 1

        # ── Registro ──────────────────────────────────────────────────────────
        report_buf[t, :, 0] = C[:, 0]
        report_buf[t, :, 1] = C[:, 1]
        report_buf[t, :, 2] = C[:, 2]
        report_buf[t, :, 3] = C[:, 3]
        report_buf[t, :, 4] = np.degrees(C[:, 3])
        report_buf[t, :, 5] = C[:, 4]
        report_buf[t, :, 6] = C[:, 5]
        report_buf[t, :, 7] = state_detected[t, :]

        obj_report_buf[t]      = objects_pos.copy()
        carrying_report_buf[t] = carrying.copy()

        if progress_callback is not None:
            n_delivered_now = int(np.sum(delivered))
            progress_callback(t, max_iter, n_delivered_now)

        t += 1

    # ── Recortar buffers ──────────────────────────────────────────────────────
    report          = report_buf[:t]
    objects_report  = obj_report_buf[:t]
    carrying_report = carrying_report_buf[:t]

    collected_objects[:, 1] = t - collected_objects[:, 0]
    collected_objects[:, 3] = t

    # ── Metricas ──────────────────────────────────────────────────────────────
    farming_result = mtr.farming_metrics(
        collected_objects = collected_objects,
        n_objects         = n_objects,
        individuals       = individuals,
        delivered         = delivered,
    )

    # ── Animacion ─────────────────────────────────────────────────────────────
    if animation:
        env = {
            "area_limits":    area_limits,
            "nest_position":  list(nest_pos),
            "nest_radius":    nest_ri + nest_rs,
            "nest_area_side": nest_area,
            "plots":          plots,
            "plot_repulsion": plot_repulsion,
            "strip_width":    strip_width,
        }
        print("Simulation complete. Starting animation...")
        viz.animate_farming(
            report          = report,
            objects_report  = objects_report,
            carrying_report = carrying_report,
            env             = env,
            interval        = config.FARMING_ANIMATION_INTERVAL,
            show_zones      = config.SHOW_ZONES,
            show_trail      = config.SHOW_TRAIL,
            trail_length    = config.TRAIL_LENGTH,
            save_path       = config.VIDEO_SAVE_PATH,
            screen_size     = config.SCREEN_SIZE,
        )

    return report, objects_report, farming_result


# ============================================================================
# Impresion de resultados
# ============================================================================

def _print_results_farming(
    metrics:     dict,
    elapsed:     float,
    iterations:  int,
    max_iter:    int,
    individuals: int,
    n_objects:   int,
    n_plots:     int,
    r_r: float, o_r: float, a_r: float,
    time_limit: Optional[int] = None,
) -> None:
    """
    Imprime tabla de resultados de una simulacion de farming.

    Reporta objetos entregados y pendientes. El loop puede terminar antes
    de max_iter si todos los objetos fueron entregados; iterations refleja
    la duracion real de la corrida.

    Args:
        metrics, elapsed, iterations, max_iter, individuals,
        n_objects, n_plots, r_r, o_r, a_r: parametros de la simulacion.
        time_limit: si se uso ventana de tiempo fija (ver farming.run),
                    se anota junto a f3 para dejar claro el modo de medicion.
    """
    W   = 54
    sep = "=" * W
    ms_iter = elapsed / max(iterations, 1) * 1000

    print(f"\n{sep}")
    print(f"  RAOI -- Farming Task - Results")
    print(sep)
    print(f"  N robots      : {individuals:>4}      Farm rows    : {n_plots:>6}")
    print(f"  Objects       : {n_objects:>4}      Iterations   : {iterations} / {max_iter}")
    print(f"  r_repulsion   : {r_r:.3f} m    r_orientation: {o_r:.3f} m")
    print(f"  r_attraction  : {a_r:.3f} m")
    print(sep)
    print(f"  f1 -- objetos por robot : {[int(x) for x in metrics['objects_per_robot']]}")
    print(f"  f1 -- balance (std)     : {metrics['load_balance_std']:>9.3f}")
    print(sep)
    print(f"  f2 -- energia por robot : {[round(float(x), 1) for x in metrics['energy_per_robot']]}")
    print(f"  f2 -- energia total     : {metrics['total_energy']:>9.1f}")
    print(f"  f2 -- energia promedio  : {metrics['mean_energy']:>9.3f}")
    print(sep)
    _f3_window = f"ventana={time_limit} iter" if time_limit is not None else "sin ventana fija"
    print(f"  f3 -- entregados/total ({_f3_window})")
    print(f"       = {metrics['success_fraction']:>9.3f}  ({metrics['delivered']}/{n_objects})")
    print(sep)
    print(f"  Runtime : {elapsed:.2f} s   ({ms_iter:.1f} ms/iter)")
    print(f"{sep}\n")


def _print_stats_farming(
    metrics_report: np.ndarray,
    elapsed:        float,
    replicas:       int,
    n_objects:      int,
) -> None:
    """
    Imprime tabla estadistica de multiples replicas de farming.

    Args:
        metrics_report: Shape (R, 3). Cols: load_balance_std (f1),
                        mean_energy (f2), success_fraction (f3).
        elapsed:        Tiempo total de ejecucion en segundos.
        replicas:       Numero de replicas.
        n_objects:      Numero de objetos por replica.
    """
    W    = 54
    sep  = "=" * W
    mean = np.mean(metrics_report, axis=0)
    std  = np.std(metrics_report,  axis=0)
    pct_complete = np.mean(metrics_report[:, 2] == 1.0) * 100

    print(f"\n{sep}")
    print(f"  RAOI -- Farming - Statistical Results ({replicas} replicas)")
    print(sep)
    print(f"  {'Metric':<30} {'Mean':>7}  {'Std':>7}")
    print(f"  {'-'*30} {'-'*7}  {'-'*7}")
    print(f"  {'load_balance_std (f1)':<30} {mean[0]:>7.3f}  {std[0]:>7.3f}")
    print(f"  {'mean_energy (f2)':<30} {mean[1]:>7.3f}  {std[1]:>7.3f}")
    print(f"  {'success_fraction (f3)':<30} {mean[2]:>7.3f}  {std[2]:>7.3f}")
    print(sep)
    print(f"  Task completion rate    : {pct_complete:.1f}% of replicas")
    print(f"  Avg runtime / replica   : {elapsed / replicas:.2f} s")
    print(f"  Total runtime           : {elapsed:.2f} s")
    print(f"{sep}\n")


# ============================================================================
# Runners
# ============================================================================

def single_run(
    n_objects:   Optional[int]   = None,
    n_plots:     Optional[int]   = None,
    individuals: Optional[int]   = None,
    r_r:         Optional[float] = None,
    o_r:         Optional[float] = None,
    a_r:         Optional[float] = None,
    animation:   Optional[bool]  = None,
    nest_ri:     Optional[float] = None,
    nest_rs:     Optional[float] = None,
    obj_r:       Optional[float] = None,
    time_limit:  Optional[int]   = None,
) -> tuple:
    """
    Ejecuta una simulacion de farming con barra de progreso y tabla de resultados.

    Solicita los parametros por consola si no se proporcionan. Muestra
    el maximo de parcelas permitido junto al prompt de n_plots.

    Args:
        n_objects, n_plots, individuals, r_r, o_r, a_r, animation:
            Parametros de la simulacion (todos opcionales, se piden por
            consola si faltan).
        nest_ri, nest_rs, obj_r, time_limit:
            Parametros avanzados (radios de deteccion y ventana de tiempo
            fija) — ver farming.run(). Se piden por consola igual que el
            resto; dejar en blanco (Enter) usa el valor por defecto de
            config.py (o config.FARMING_DEFAULT_MAX_ITER, para time_limit).

    Returns:
        (report, objects_report, metrics) -- idem farming.run().
    """
    def _int(prompt, val):
        if val is not None:
            return val
        while True:
            try:
                return int(input(prompt))
            except ValueError:
                print("  Invalid input. Enter an integer.")

    def _float(prompt, val):
        if val is not None:
            return val
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("  Invalid input. Enter a number (e.g. 1.0).")

    def _bool(prompt, val):
        if val is not None:
            return val
        while True:
            ans = input(prompt).strip().upper()
            if ans in ("YES", "NO"):
                return ans == "YES"
            print("  Enter YES or NO.")

    def _float_opt(prompt, val, default):
        """Como _float, pero Enter en blanco devuelve None (usa default de config.py)."""
        if val is not None:
            return val
        while True:
            raw = input(f"{prompt} [Enter = {default}]: ").strip()
            if raw == "":
                return None
            try:
                return float(raw)
            except ValueError:
                print("  Invalid input. Enter a number or leave blank.")

    def _int_opt(prompt, val):
        """Como _int, pero Enter en blanco devuelve None (usa el valor por defecto)."""
        if val is not None:
            return val
        while True:
            raw = input(prompt).strip()
            if raw == "":
                return None
            try:
                return int(raw)
            except ValueError:
                print("  Invalid input. Enter an integer or leave blank.")

    max_plots = _compute_max_plots(
        config.FARMING_AREA_LIMITS, config.FARMING_PLOT_SEPARATION,
        config.FARMING_STRIP_WIDTH, config.FARMING_PLOT_REPULSION,
        config.FARMING_PLOT_MARGIN_Y,
    )

    n_objects   = _int  ("Objects: ",                                        n_objects)
    n_plots     = _int  (f"Number of farm rows (plots) [max {max_plots}]: ", n_plots)
    individuals = _int  ("Individuals: ",                                    individuals)
    r_r         = _float("Repulsion radius (m): ",                           r_r)
    o_r         = _float("Orientation radius (m): ",                         o_r)
    a_r         = _float("Attraction radius (m): ",                          a_r)
    nest_ri     = _float_opt("Nest detection radius ri (m)",   nest_ri, config.FARMING_NEST_RI)
    nest_rs     = _float_opt("Nest emission radius rs (m)",    nest_rs, config.FARMING_NEST_RS)
    obj_r       = _float_opt("Object detection radius (m)",    obj_r,   config.FARMING_OBJECT_DETECTION_RADIUS)
    time_limit  = _int_opt  ("Fixed time window in iterations (blank = default max_iter): ", time_limit)
    animation   = _bool ("Animation? (YES/NO): ",                            animation)

    max_iter = int(time_limit) if time_limit is not None else config.FARMING_DEFAULT_MAX_ITER
    print(f"  Max iterations: {max_iter}" + (" (time_limit)" if time_limit is not None else " (default)"))

    bar = tqdm(
        total  = max_iter,
        desc   = "Farming   ",
        unit   = "iter",
        ncols  = 72,
        colour = "green",
    )
    bar.set_postfix(obj=f"0/{n_objects}")
    last_t = [0]

    def _cb(t: int, total: int, delivered: int) -> None:
        bar.update(t - last_t[0])
        bar.set_postfix(obj=f"{delivered}/{n_objects}")
        last_t[0] = t

    t0 = time.time()
    report, objects_report, metrics = run(
        n_objects, n_plots, individuals, r_r, o_r, a_r, animation,
        nest_ri=nest_ri, nest_rs=nest_rs, obj_r=obj_r, time_limit=time_limit,
        progress_callback=_cb,
    )
    bar.update(report.shape[0] - last_t[0])
    bar.set_postfix(obj=f"{metrics['delivered']}/{n_objects}")
    bar.close()
    elapsed = time.time() - t0

    _print_results_farming(
        metrics, elapsed, report.shape[0], max_iter,
        individuals, n_objects, n_plots, r_r, o_r, a_r,
        time_limit=time_limit,
    )

    np.save("farming_report",         report)
    np.save("farming_objects_report", objects_report)

    return report, objects_report, metrics


def statistical_run(
    replicas:    int,
    n_objects:   Optional[int]   = None,
    n_plots:     Optional[int]   = None,
    individuals: Optional[int]   = None,
    r_r:         Optional[float] = None,
    o_r:         Optional[float] = None,
    a_r:         Optional[float] = None,
    nest_ri:     Optional[float] = None,
    nest_rs:     Optional[float] = None,
    obj_r:       Optional[float] = None,
    time_limit:  Optional[int]   = None,
) -> tuple:
    """
    Ejecuta multiples replicas de farming con barra de progreso y tabla estadistica.

    Cada replica usa config.SEED + replica_index como semilla.

    Args:
        replicas:    Numero de replicas a ejecutar.
        n_objects, n_plots, individuals, r_r, o_r, a_r: idem single_run().
        nest_ri, nest_rs, obj_r, time_limit: idem farming.run() — se piden
            por consola igual que el resto; dejar en blanco (Enter) usa el
            valor por defecto de config.py (o config.FARMING_DEFAULT_MAX_ITER,
            para time_limit). Se aplican igual a todas las replicas.

    Returns:
        metrics_report : Metricas por replica, shape (R, 3). Columnas:
                         [load_balance_std (f1), mean_energy (f2),
                          success_fraction (f3)].
        running_mean   : Promedio acumulado, shape (R, 3).
        final_mean     : Media global, shape (3,).
    """
    def _int(p, v):
        if v is not None:
            return v
        while True:
            try:
                return int(input(p))
            except ValueError:
                print("  Invalid input. Enter an integer.")

    def _float(p, v):
        if v is not None:
            return v
        while True:
            try:
                return float(input(p))
            except ValueError:
                print("  Invalid input. Enter a number (e.g. 1.0).")

    def _float_opt(prompt, val, default):
        """Como _float, pero Enter en blanco devuelve None (usa default de config.py)."""
        if val is not None:
            return val
        while True:
            raw = input(f"{prompt} [Enter = {default}]: ").strip()
            if raw == "":
                return None
            try:
                return float(raw)
            except ValueError:
                print("  Invalid input. Enter a number or leave blank.")

    def _int_opt(prompt, val):
        """Como _int, pero Enter en blanco devuelve None (usa el valor por defecto)."""
        if val is not None:
            return val
        while True:
            raw = input(prompt).strip()
            if raw == "":
                return None
            try:
                return int(raw)
            except ValueError:
                print("  Invalid input. Enter an integer or leave blank.")

    max_plots = _compute_max_plots(
        config.FARMING_AREA_LIMITS, config.FARMING_PLOT_SEPARATION,
        config.FARMING_STRIP_WIDTH, config.FARMING_PLOT_REPULSION,
        config.FARMING_PLOT_MARGIN_Y,
    )

    n_objects   = _int  ("Objects: ",                                        n_objects)
    n_plots     = _int  (f"Number of farm rows (plots) [max {max_plots}]: ", n_plots)
    individuals = _int  ("Individuals: ",                                    individuals)
    r_r         = _float("Repulsion radius (m): ",                           r_r)
    o_r         = _float("Orientation radius (m): ",                         o_r)
    a_r         = _float("Attraction radius (m): ",                          a_r)
    nest_ri     = _float_opt("Nest detection radius ri (m)",   nest_ri, config.FARMING_NEST_RI)
    nest_rs     = _float_opt("Nest emission radius rs (m)",    nest_rs, config.FARMING_NEST_RS)
    obj_r       = _float_opt("Object detection radius (m)",    obj_r,   config.FARMING_OBJECT_DETECTION_RADIUS)
    time_limit  = _int_opt  ("Fixed time window in iterations (blank = default max_iter): ", time_limit)

    max_iter = int(time_limit) if time_limit is not None else config.FARMING_DEFAULT_MAX_ITER

    metrics_report = np.zeros((replicas, 3))
    running_mean   = np.zeros((replicas, 3))

    rep_bar = tqdm(
        total  = replicas,
        desc   = "Replicas  ",
        unit   = "rep",
        ncols  = 72,
        colour = "cyan",
    )
    rep_bar.set_postfix(obj=f"0/{n_objects}")

    t0 = time.time()

    for r in range(replicas):
        iter_bar = tqdm(
            total  = max_iter,
            desc   = f"  Rep {r+1:>3}/{replicas}",
            unit   = "iter",
            ncols  = 72,
            leave  = False,
            colour = "green",
        )
        last_t = [0]

        def _cb(t: int, total: int, delivered: int, _b=iter_bar, _l=last_t) -> None:
            _b.update(t - _l[0])
            _b.set_postfix(obj=f"{delivered}/{n_objects}")
            _l[0] = t

        _, _, metrics = run(
            n_objects, n_plots, individuals, r_r, o_r, a_r,
            animation=False,
            seed=config.SEED + r,
            nest_ri=nest_ri, nest_rs=nest_rs, obj_r=obj_r, time_limit=time_limit,
            progress_callback=_cb,
        )
        iter_bar.close()

        metrics_report[r] = [
            metrics["load_balance_std"], metrics["mean_energy"],
            metrics["success_fraction"],
        ]
        rep_bar.set_postfix(
            obj=f"{metrics['delivered']}/{n_objects}",
        )
        rep_bar.update(1)

    rep_bar.close()
    elapsed = time.time() - t0

    final_mean = np.mean(metrics_report, axis=0)
    for r in range(replicas):
        running_mean[r] = np.mean(metrics_report[:r+1], axis=0)

    _print_stats_farming(metrics_report, elapsed, replicas, n_objects)

    np.save("farming_metrics_report", metrics_report)
    np.save("farming_running_mean",   running_mean)
    np.save("farming_final_mean",     final_mean)

    return metrics_report, running_mean, final_mean