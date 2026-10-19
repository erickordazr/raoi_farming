"""Visualización de la tarea de farming.

Reproduce cuadro a cuadro (Pygame) una simulación ya calculada: robots,
objetos, parcelas y nest. Puede mostrarse en una ventana o grabarse en un
video mp4 (OpenCV).

Capas opcionales (config.py):
  SHOW_ZONES         → zonas RAOI de cada robot
  SHOW_TRAIL         → rastro de los últimos TRAIL_LENGTH pasos
  SHOW_SENSOR_RADIUS → sensor activo de cada robot: cono de objetos r_d
                       (sin carga) o círculo de nest r_i (cargado)

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import logging
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import pygame
from tqdm import tqdm

from . import config
from .behavior import Zone

logger = logging.getLogger(__name__)

# ── Paleta ────────────────────────────────────────────────────────────────────

BG_COLOR            = (255, 255, 255)
BORDER_COLOR        = ( 40,  40,  40)
GRID_COLOR          = (220, 220, 220)
TEXT_COLOR          = ( 30,  30,  30)
TRAIL_ALPHA         = 120                  # opacidad máxima del rastro (0-255)

PLOT_COLOR          = ( 60, 120,  50)      # borde y extremos de la parcela
PLOT_BED_COLOR      = ( 90, 145,  70)      # cama de cultivo
PLOT_FILL_COLOR     = (200, 230, 190)      # franja de objetos
PLOT_REPULSION_RGBA = (200, 100,  50, 30)  # zona de repulsión de la parcela
FRUIT_COLOR         = (220,  80,  30)      # objeto en el suelo
FRUIT_CARRIED       = (255, 140,  60)      # borde del objeto cargado
CORRIDOR_COLOR      = (222, 210, 180)      # pasillo oeste
NEST_INFLUENCE_RGBA = (100, 180, 100, 55)  # radio de emisión r_s del nest
NEST_MARK_COLOR     = ( 40, 110,  40)      # borde del radio de emisión y etiqueta NEST
NEST_SENSOR_FILL_RGBA = ( 70, 130, 200, 10)   # relleno del sensor de nest r_i (robot cargado)
NEST_SENSOR_RGBA      = ( 70, 130, 200, 170)  # borde del sensor de nest r_i
OBJ_SENSOR_FILL_RGBA  = (240, 150,  30, 90)   # relleno del cono del sensor r_d (robot sin carga)
OBJ_SENSOR_RGBA       = (225, 120,  20, 230)  # borde del cono del sensor de objetos
DEPOSIT_COLOR       = ( 30, 100,  30)      # círculo de depósito y borde de objetos entregados
DELIVERED_COLOR     = ( 60, 160,  60)      # objeto entregado

# Color por zona activa (columna 7 del reporte, ver behavior.Zone).
STATE_RGB = {
    Zone.FREE:        (160, 160, 160),   # exploración libre — gris
    Zone.REPULSION:   (220,  50,  60),   # repulsión         — rojo
    Zone.ATTRACTION:  ( 60, 120, 170),   # atracción         — azul
    Zone.ORIENTATION: ( 38, 160, 140),   # orientación       — verde
    Zone.INFLUENCE:   (220, 185,  80),   # influencia        — dorado
}
STATE_LABELS = {
    Zone.FREE:        "Free exploration",
    Zone.REPULSION:   "Repulsion",
    Zone.ATTRACTION:  "Attraction",
    Zone.ORIENTATION: "Orientation",
    Zone.INFLUENCE:   "Influence",
}
ZONE_RGBA = {
    "repulsion":   (220,  50,  60, 35),
    "orientation": ( 38, 160, 140, 20),
    "attraction":  ( 60, 120, 170, 15),
}


# ── Geometría del icono del robot ─────────────────────────────────────────────

def _rotate_points(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rota puntos alrededor del origen.

    Args:
        pts:   Puntos, shape (N, 2).
        angle: Ángulo de rotación (rad).

    Returns:
        Puntos rotados, shape (N, 2).
    """
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return pts @ rotation.T


def _to_int_points(pts: list) -> list:
    """Convierte una lista de puntos [x, y] a tuplas de enteros para Pygame."""
    return [(int(x), int(y)) for x, y in pts]


def _robot_polygons(cx: float, cy: float, theta: float, r: float) -> dict:
    """Vértices del icono del robot diferencial, en píxeles.

    Cuerpo circular, dos ruedas laterales, nariz triangular al frente y una
    flecha que sale de la nariz en la dirección de avance.

    Args:
        cx:     Coordenada X del centro del robot (px).
        cy:     Coordenada Y del centro del robot (px).
        theta:  Rumbo del robot en coordenadas de simulación (rad).
        r:      Radio del cuerpo (px).

    Returns:
        Dict con 'body_center' (cx, cy, r), 'nose', 'wheel_l' y 'wheel_r'
        (listas de vértices) y 'arrow' (inicio, fin).
    """
    nose_pts = np.array([
        [r * 1.45,  0.0      ],   # punta
        [r * 0.80,  r * 0.38],    # base derecha
        [r * 0.80, -r * 0.38],    # base izquierda
    ])

    wl = r * 1.10   # largo de la rueda
    wh = r * 0.45   # ancho de la rueda
    wy = r * 1.00   # distancia lateral al borde interno de la rueda

    wl_pts = np.array([[-wl/2,  wy], [wl/2,  wy], [wl/2,  wy + wh], [-wl/2,  wy + wh]])
    wr_pts = np.array([[-wl/2, -wy - wh], [wl/2, -wy - wh], [wl/2, -wy], [-wl/2, -wy]])

    arrow_start = np.array([[r * 1.45, 0.0]])
    arrow_end   = np.array([[r * 2.10, 0.0]])

    # En pantalla el eje Y está invertido: se rota con -theta.
    angle  = -theta
    center = np.array([cx, cy])

    return {
        "body_center": (cx, cy, r),
        "nose":        (_rotate_points(nose_pts, angle) + center).tolist(),
        "wheel_l":     (_rotate_points(wl_pts,   angle) + center).tolist(),
        "wheel_r":     (_rotate_points(wr_pts,   angle) + center).tolist(),
        "arrow":       ((_rotate_points(arrow_start, angle) + center)[0],
                        (_rotate_points(arrow_end,   angle) + center)[0]),
    }


# ── Conversión mundo → pantalla ───────────────────────────────────────────────

class WorldToScreen:
    """Convierte coordenadas de simulación (m, Y hacia arriba) a píxeles (Y hacia abajo)."""

    def __init__(self, area_m: float, screen_px: int, margin_px: int = 40) -> None:
        """Calcula la escala a partir del área simulada y del tamaño de la superficie.

        Args:
            area_m:    Lado del área simulada (m).
            screen_px: Lado de la superficie de dibujo (px).
            margin_px: Margen alrededor del área (px).
        """
        self.area_m = area_m                                   # lado del área (m)
        self.margin = margin_px                                # margen (px)
        self.scale  = (screen_px - 2 * margin_px) / area_m     # escala (px/m)

    def xy(self, xm: float, ym: float) -> tuple[int, int]:
        """Posición (m) → píxel."""
        px = int(self.margin + xm * self.scale)
        py = int(self.margin + (self.area_m - ym) * self.scale)
        return px, py

    def r(self, rm: float) -> int:
        """Longitud (m) → píxeles (al menos 1)."""
        return max(1, int(rm * self.scale))


# ── Dibujo del escenario ──────────────────────────────────────────────────────

def _draw_farming_environment(
    surf: pygame.Surface,
    w2s: WorldToScreen,
    env: dict,
) -> None:
    """Dibuja el pasillo oeste, el nest, las franjas de objetos y las parcelas.

    Args:
        surf: Superficie de dibujo.
        w2s:  Conversor mundo → pantalla.
        env:  Escenario (ver animate_farming).
    """
    plots          = env["plots"]
    area_limits    = env["area_limits"]
    strip_width    = env["strip_width"]
    plot_repulsion = env["plot_repulsion"]
    margin         = plot_repulsion + env["strip_gap"]   # inicio de la franja desde el eje

    # ── Pasillo oeste: de la pared oeste al inicio de las parcelas ────────────
    corridor_x1 = min(float(p["x0"]) for p in plots) if plots else area_limits
    if corridor_x1 > 0.0:
        c_lo = w2s.xy(0.0, area_limits)
        c_hi = w2s.xy(corridor_x1, 0.0)
        corridor_s = pygame.Surface((max(1, c_hi[0] - c_lo[0]), max(1, c_hi[1] - c_lo[1])),
                                    pygame.SRCALPHA)
        corridor_s.fill((*CORRIDOR_COLOR, 90))
        surf.blit(corridor_s, c_lo)

    # ── Nest ──────────────────────────────────────────────────────────────────
    # Radio de emisión r_s relleno y con borde, círculo de depósito y
    # etiqueta. Un robot percibe el nest cuando su radio sensor r_i (dibujado
    # en cada robot) alcanza este círculo: distancia < r_i + r_s.
    nc_px = w2s.xy(env["nest_position"][0], env["nest_position"][1])
    r_px  = w2s.r(env["nest_rs"])
    circle_surf = pygame.Surface((r_px * 2 + 2, r_px * 2 + 2), pygame.SRCALPHA)
    pygame.draw.circle(circle_surf, NEST_INFLUENCE_RGBA, (r_px + 1, r_px + 1), r_px)
    surf.blit(circle_surf, (nc_px[0] - r_px - 1, nc_px[1] - r_px - 1))
    pygame.draw.circle(surf, NEST_MARK_COLOR, nc_px, r_px, 2)

    pygame.draw.circle(surf, DEPOSIT_COLOR, nc_px, max(2, w2s.r(env["deposit_radius"])), 1)

    label_gap_px = max(6, w2s.r(env["nest_icon_radius"]))   # separación de la etiqueta
    font_nest = pygame.font.SysFont("Arial", 11, bold=True)
    lbl = font_nest.render("NEST", True, NEST_MARK_COLOR)
    surf.blit(lbl, (nc_px[0] - lbl.get_width() // 2,
                    nc_px[1] - label_gap_px - lbl.get_height() - 4))

    # ── Franjas de objetos (arriba y abajo de cada parcela) ───────────────────
    for plot in plots:
        x0, x1, yp = plot["x0"], plot["x1"], plot["y"]
        for yf_lo, yf_hi in ((yp + margin, yp + margin + strip_width),
                             (yp - margin - strip_width, yp - margin)):
            px_lo = w2s.xy(x0, yf_hi)
            px_hi = w2s.xy(x1, yf_lo)
            fs = pygame.Surface((max(1, px_hi[0] - px_lo[0]), max(1, px_hi[1] - px_lo[1])),
                                pygame.SRCALPHA)
            fs.fill((*PLOT_FILL_COLOR, 60))
            surf.blit(fs, px_lo)

    # ── Parcelas: zona de repulsión, cama de cultivo y extremos ───────────────
    plot_rep_px = w2s.r(plot_repulsion)
    bed_half_px = max(1, w2s.r(env["plot_visual_thickness"]))

    for plot in plots:
        p0 = w2s.xy(plot["x0"], plot["y"])
        p1 = w2s.xy(plot["x1"], plot["y"])
        width_px = p1[0] - p0[0]

        rep_s = pygame.Surface((max(1, width_px), max(1, plot_rep_px * 2)), pygame.SRCALPHA)
        rep_s.fill(PLOT_REPULSION_RGBA)
        surf.blit(rep_s, (p0[0], p0[1] - plot_rep_px))

        bed_rect = (p0[0], p0[1] - bed_half_px, max(1, width_px), max(2, bed_half_px * 2))
        pygame.draw.rect(surf, PLOT_BED_COLOR, bed_rect)
        pygame.draw.rect(surf, PLOT_COLOR, bed_rect, 1)

        for px_pt in (p0, p1):
            pygame.draw.line(surf, PLOT_COLOR,
                             (px_pt[0], px_pt[1] - plot_rep_px),
                             (px_pt[0], px_pt[1] + plot_rep_px), 2)


def _draw_farming_objects(
    surf: pygame.Surface,
    w2s: WorldToScreen,
    obj_positions: np.ndarray,
    carried_set: set,
    nest_pos: list,
    deposit_radius: float,
) -> None:
    """Dibuja los objetos que no están cargados.

    Los objetos en el suelo van en naranja y los entregados, dentro del
    círculo de depósito, en verde. Los cargados se dibujan sobre el robot.

    Args:
        surf:           Superficie de dibujo.
        w2s:            Conversor mundo → pantalla.
        obj_positions:  Posiciones de los objetos, shape (O, 2).
        carried_set:    Índices de los objetos cargados en este cuadro.
        nest_pos:       Posición [x, y] del nest.
        deposit_radius: Radio de depósito (m).
    """
    obj_r = max(5, w2s.r(0.15))   # radio del icono (px)

    for o, pos in enumerate(obj_positions):
        if o in carried_set:
            continue

        cx, cy   = w2s.xy(pos[0], pos[1])
        dxo, dyo = pos[0] - nest_pos[0], pos[1] - nest_pos[1]

        if (dxo * dxo + dyo * dyo) <= deposit_radius * deposit_radius:
            pygame.draw.circle(surf, DELIVERED_COLOR, (cx, cy), max(3, obj_r - 2))
            pygame.draw.circle(surf, DEPOSIT_COLOR,   (cx, cy), max(3, obj_r - 2), 1)
        else:
            pygame.draw.circle(surf, (90, 45, 0),     (cx + 2, cy + 2), obj_r)   # sombra
            pygame.draw.circle(surf, FRUIT_COLOR,     (cx, cy), obj_r)
            pygame.draw.circle(surf, (255, 200, 100), (cx, cy), obj_r, 1)


def _draw_farming_hud(
    surf: pygame.Surface,
    font_sm: pygame.font.Font,
    font_title: pygame.font.Font,
    frame: int,
    iterations: int,
    n_robots: int,
    n_objects: int,
    n_plots: int,
    delivered: int,
) -> None:
    """Dibuja título, número de cuadro, barra de entregas y datos del escenario.

    Args:
        surf:       Superficie de dibujo.
        font_sm:    Fuente del texto normal.
        font_title: Fuente del título.
        frame:      Cuadro actual (desde 0).
        iterations: Total de cuadros.
        n_robots:   Número de robots.
        n_objects:  Número de objetos.
        n_plots:    Número de parcelas.
        delivered:  Objetos entregados en este cuadro.
    """
    width, height = surf.get_size()

    title = font_title.render("RAOI — Farming Task", True, TEXT_COLOR)
    surf.blit(title, (width // 2 - title.get_width() // 2, 6))

    frame_txt = font_sm.render(f"Frame {frame + 1} / {iterations}", True, TEXT_COLOR)
    surf.blit(frame_txt, (width - frame_txt.get_width() - 10, 6))

    bar_w, bar_h = 160, 14
    bar_x, bar_y = width - bar_w - 10, 26
    frac = delivered / max(n_objects, 1)
    pygame.draw.rect(surf, (220, 220, 220), (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(surf, FRUIT_COLOR,     (bar_x, bar_y, int(bar_w * frac), bar_h))
    pygame.draw.rect(surf, (80, 80, 80),    (bar_x, bar_y, bar_w, bar_h), 1)
    prog_txt = font_sm.render(f"{delivered}/{n_objects} fruits", True, TEXT_COLOR)
    surf.blit(prog_txt, (bar_x - prog_txt.get_width() - 6,
                         bar_y + (bar_h - prog_txt.get_height()) // 2))

    info = font_sm.render(
        f"Robots: {n_robots}   Plots: {n_plots}   Objects: {n_objects}",
        True, TEXT_COLOR,
    )
    surf.blit(info, (10, height - info.get_height() - 6))


# ── Capas de cada cuadro ──────────────────────────────────────────────────────

def _render_static_layer(w2s: WorldToScreen, env: dict, screen_size: int) -> pygame.Surface:
    """Dibuja una sola vez lo que no cambia entre cuadros.

    Fondo, cuadrícula de 1 m, borde del área, pasillo oeste, nest, franjas y
    parcelas.

    Args:
        w2s:         Conversor mundo → pantalla.
        env:         Escenario (ver animate_farming).
        screen_size: Lado de la superficie (px).

    Returns:
        Superficie opaca con la capa estática.
    """
    area  = env["area_limits"]
    layer = pygame.Surface((screen_size, screen_size))
    layer.fill(BG_COLOR)
    v = 0.0
    while v <= area:
        pygame.draw.line(layer, GRID_COLOR, w2s.xy(v, 0), w2s.xy(v, area), 1)
        pygame.draw.line(layer, GRID_COLOR, w2s.xy(0, v), w2s.xy(area, v), 1)
        v += 1.0
    bx0, by0 = w2s.xy(0, area)
    bx1, by1 = w2s.xy(area, 0)
    pygame.draw.rect(layer, BORDER_COLOR, (bx0, by0, bx1 - bx0, by1 - by0), 2)
    _draw_farming_environment(layer, w2s, env)
    return layer


def _draw_trails(
    screen:    pygame.Surface,
    w2s:       WorldToScreen,
    trails:    list[deque[tuple[int, int]]],
    positions: np.ndarray,
) -> None:
    """Agrega la posición actual al rastro de cada robot y lo dibuja desvaneciéndose.

    Args:
        screen:    Superficie de dibujo.
        w2s:       Conversor mundo → pantalla.
        trails:    Últimos píxeles de cada robot (se actualizan).
        positions: Posiciones de los robots en este cuadro, shape (N, 2).
    """
    trail_s = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for i, trail in enumerate(trails):
        trail.append(w2s.xy(positions[i, 0], positions[i, 1]))
        pts_t = list(trail)
        n_pts = len(pts_t)
        for k in range(n_pts - 1):
            alpha = int((k + 1) / n_pts * TRAIL_ALPHA)
            pygame.draw.line(trail_s, (130, 130, 130, alpha), pts_t[k], pts_t[k + 1], 2)
    screen.blit(trail_s, (0, 0))


def _sector_points(
    center: tuple[float, float],
    radius: float,
    theta:  float,
    fov:    float,
    steps:  int,
) -> list[tuple[float, float]]:
    """Vértices de un sector circular centrado en el rumbo del robot.

    El eje Y de la pantalla está invertido, por eso el ángulo se usa con signo
    negativo.

    Args:
        center: Centro del sector (px).
        radius: Radio del sector (px).
        theta:  Rumbo del robot en coordenadas de simulación (rad).
        fov:    Apertura del sector (rad).
        steps:  Número de segmentos del arco.

    Returns:
        Centro seguido de los steps + 1 puntos del arco.
    """
    start_a = -theta - fov / 2
    return [center] + [
        (center[0] + radius * math.cos(start_a + fov * k / steps),
         center[1] + radius * math.sin(start_a + fov * k / steps))
        for k in range(steps + 1)
    ]


def _draw_sensors(
    screen:         pygame.Surface,
    w2s:            WorldToScreen,
    positions:      np.ndarray,
    orientations:   np.ndarray,
    loaded:         np.ndarray,
    nest_sensor_px: int,
    obj_sensor_px:  int,
    obj_sensor_fov: float,
) -> None:
    """Dibuja el sensor activo de cada robot.

    Sin carga, el cono del sensor de objetos; cargado, el círculo del sensor
    de nest.

    Args:
        screen:         Superficie de dibujo.
        w2s:            Conversor mundo → pantalla.
        positions:      Posiciones de los robots, shape (N, 2).
        orientations:   Rumbos de los robots (rad), shape (N,).
        loaded:         True si el robot carga un objeto, shape (N,).
        nest_sensor_px: Radio del sensor de nest (px).
        obj_sensor_px:  Alcance del sensor de objetos (px).
        obj_sensor_fov: Apertura del sensor de objetos (rad).
    """
    sensor_s = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    steps    = max(12, int(math.degrees(obj_sensor_fov) / 5))
    for i in range(len(positions)):
        c_px = w2s.xy(positions[i, 0], positions[i, 1])
        if loaded[i]:
            pygame.draw.circle(sensor_s, NEST_SENSOR_FILL_RGBA, c_px, nest_sensor_px)
            pygame.draw.circle(sensor_s, NEST_SENSOR_RGBA, c_px, nest_sensor_px, 2)
        else:
            cone = _sector_points(c_px, obj_sensor_px, orientations[i], obj_sensor_fov, steps)
            pygame.draw.polygon(sensor_s, OBJ_SENSOR_FILL_RGBA, cone)
            pygame.draw.polygon(sensor_s, OBJ_SENSOR_RGBA, cone, 1)
    screen.blit(sensor_s, (0, 0))


def _draw_zones(
    screen: pygame.Surface,
    center: tuple[int, int],
    theta:  float,
    radii:  tuple[int, int, int],
) -> None:
    """Dibuja las zonas de repulsión, orientación y atracción de un robot.

    Args:
        screen: Superficie de dibujo.
        center: Centro del robot (px).
        theta:  Rumbo del robot (rad).
        radii:  Radios de repulsión, orientación y atracción (px).
    """
    cx, cy = center
    for r_px, rgba, fov_key in (
        (radii[0], ZONE_RGBA["repulsion"],   "fov_repulsion"),
        (radii[1], ZONE_RGBA["orientation"], "fov_orientation"),
        (radii[2], ZONE_RGBA["attraction"],  "fov_attraction"),
    ):
        fov_v  = config.RAOI_FOV[fov_key]
        zone_s = pygame.Surface((r_px * 2 + 2, r_px * 2 + 2), pygame.SRCALPHA)
        if fov_v >= 2 * math.pi - 0.01:
            pygame.draw.circle(zone_s, rgba, (r_px + 1, r_px + 1), r_px)
        else:
            steps = max(20, int(math.degrees(fov_v)))
            pts   = _sector_points((r_px + 1, r_px + 1), r_px, theta, fov_v, steps)
            pygame.draw.polygon(zone_s, rgba, pts)
        screen.blit(zone_s, (cx - r_px - 1, cy - r_px - 1))


def _draw_robot(
    screen:  pygame.Surface,
    font:    pygame.font.Font,
    index:   int,
    center:  tuple[int, int],
    theta:   float,
    color:   tuple[int, int, int],
    body_px: int,
    loaded:  bool,
) -> None:
    """Dibuja el icono de un robot, el objeto que carga y su número.

    Args:
        screen:  Superficie de dibujo.
        font:    Fuente del número del robot.
        index:   Índice del robot.
        center:  Centro del robot (px).
        theta:   Rumbo del robot (rad).
        color:   Color del cuerpo según la zona activa.
        body_px: Radio del icono (px).
        loaded:  True si el robot carga un objeto.
    """
    cx, cy = center
    polys = _robot_polygons(cx, cy, theta, body_px)
    bx, by, br = polys["body_center"]

    pygame.draw.circle(screen, (185, 185, 185), (bx + 2, by + 2), br)   # sombra
    for wkey in ("wheel_l", "wheel_r"):
        pygame.draw.polygon(screen, (35, 35, 35), _to_int_points(polys[wkey]))
        pygame.draw.polygon(screen, (60, 60, 60), _to_int_points(polys[wkey]), 1)

    pygame.draw.circle(screen, color, (bx, by), br)
    pygame.draw.circle(screen, (25, 25, 25), (bx, by), br, 1)

    nose_color = tuple(min(255, c + 45) for c in color)
    pygame.draw.polygon(screen, nose_color,   _to_int_points(polys["nose"]))
    pygame.draw.polygon(screen, (25, 25, 25), _to_int_points(polys["nose"]), 1)

    hub_r = max(2, br // 5)
    pygame.draw.circle(screen, (245, 245, 245), (bx, by), hub_r)
    pygame.draw.circle(screen, (40, 40, 40),    (bx, by), hub_r, 1)

    arr_s, arr_e = polys["arrow"]
    pygame.draw.line(screen, (15, 15, 15),
                     (int(arr_s[0]), int(arr_s[1])), (int(arr_e[0]), int(arr_e[1])), 2)
    tip_x, tip_y = int(arr_e[0]), int(arr_e[1])
    tip_size = max(5, br // 2)
    pygame.draw.polygon(screen, (15, 15, 15), [
        (tip_x, tip_y),
        (int(tip_x - tip_size * math.cos(-theta - 0.42)),
         int(tip_y - tip_size * math.sin(-theta - 0.42))),
        (int(tip_x - tip_size * math.cos(-theta + 0.42)),
         int(tip_y - tip_size * math.sin(-theta + 0.42))),
    ])

    # Objeto cargado, dibujado en la nariz del robot.
    if loaded:
        nose_x = int(bx + body_px * 1.45 * math.cos(-theta))
        nose_y = int(by + body_px * 1.45 * math.sin(-theta))
        load_r = max(5, body_px // 2)
        pygame.draw.circle(screen, (120, 50, 0),  (nose_x + 2, nose_y + 2), load_r)
        pygame.draw.circle(screen, FRUIT_COLOR,   (nose_x, nose_y), load_r)
        pygame.draw.circle(screen, FRUIT_CARRIED, (nose_x, nose_y), load_r, 2)

    if config.SHOW_ROBOT_IDS:
        id_surf = font.render(str(index), True, (20, 20, 20))
        id_x    = cx - id_surf.get_width() // 2
        id_y    = cy - br - id_surf.get_height() - 1
        id_bg   = pygame.Surface((id_surf.get_width() + 4, id_surf.get_height() + 2),
                                 pygame.SRCALPHA)
        id_bg.fill((255, 255, 255, 160))
        screen.blit(id_bg,   (id_x - 2, id_y - 1))
        screen.blit(id_surf, (id_x, id_y))


def _draw_legend(screen: pygame.Surface, font: pygame.font.Font) -> None:
    """Dibuja la leyenda de colores por zona activa.

    Args:
        screen: Superficie de dibujo.
        font:   Fuente de las etiquetas.
    """
    leg_x, leg_y = 10, 44
    leg_surf = pygame.Surface((200, len(STATE_RGB) * 22 + 10), pygame.SRCALPHA)
    leg_surf.fill((255, 255, 255, 200))
    screen.blit(leg_surf, (leg_x - 4, leg_y - 4))
    for state_id, label in STATE_LABELS.items():
        pygame.draw.circle(screen, STATE_RGB[state_id], (leg_x + 8, leg_y + 8), 7)
        pygame.draw.circle(screen, (30, 30, 30),        (leg_x + 8, leg_y + 8), 7, 1)
        screen.blit(font.render(label, True, TEXT_COLOR), (leg_x + 20, leg_y + 1))
        leg_y += 22


@dataclass(frozen=True)
class _RobotLayers:
    """Tamaños en píxeles y capas opcionales con que se dibuja cada robot.

    Attributes:
        body_px:        Radio del icono del robot (px).
        zones_search:   Radios (R, O, A) sin carga (px); None = no dibujar zonas.
        zones_deliver:  Radios (R, O, A) con carga (px); None = no dibujar zonas.
        show_sensor:    Dibujar el sensor activo de cada robot.
        nest_sensor_px: Radio del sensor de nest r_i (px).
        obj_sensor_px:  Alcance del sensor de objetos (px).
        obj_sensor_fov: Apertura del sensor de objetos (rad).
    """

    body_px:        int
    zones_search:   tuple[int, int, int] | None
    zones_deliver:  tuple[int, int, int] | None
    show_sensor:    bool
    nest_sensor_px: int
    obj_sensor_px:  int
    obj_sensor_fov: float

    @classmethod
    def from_env(
        cls, w2s: WorldToScreen, env: dict, show_zones: bool, show_sensor: bool
    ) -> "_RobotLayers":
        """Convierte los radios del escenario a píxeles.

        Args:
            w2s:         Conversor mundo → pantalla.
            env:         Escenario (ver animate_farming).
            show_zones:  Dibujar las zonas RAOI.
            show_sensor: Dibujar el sensor activo.

        Returns:
            Capas de dibujo de los robots.
        """
        def to_px(radii: tuple[float, float, float]) -> tuple[int, int, int]:
            return w2s.r(radii[0]), w2s.r(radii[1]), w2s.r(radii[2])

        return cls(
            body_px        = max(6, w2s.r(config.ROBOT_BODY_RADIUS * config.ROBOT_VISUAL_SCALE)),
            zones_search   = to_px(env["raoi_radii_search"]) if show_zones else None,
            zones_deliver  = to_px(env["raoi_radii_deliver"]) if show_zones else None,
            show_sensor    = show_sensor,
            nest_sensor_px = w2s.r(env["nest_ri"]),
            obj_sensor_px  = w2s.r(env["obj_sensor_range"]),
            obj_sensor_fov = env["obj_sensor_fov"],
        )


def _draw_swarm(
    screen: pygame.Surface,
    w2s:    WorldToScreen,
    font:   pygame.font.Font,
    state:  np.ndarray,
    loaded: np.ndarray,
    trails: list[deque[tuple[int, int]]] | None,
    layers: _RobotLayers,
) -> None:
    """Dibuja rastros, sensores, zonas e iconos de todos los robots en un cuadro.

    Args:
        screen: Superficie de dibujo.
        w2s:    Conversor mundo → pantalla.
        font:   Fuente del número de cada robot.
        state:  Fila del reporte para este cuadro, shape (N, 8).
        loaded: True si el robot carga un objeto, shape (N,).
        trails: Rastros de los robots (se actualizan); None = no dibujar rastros.
        layers: Tamaños y capas opcionales.
    """
    positions    = state[:, :2]
    orientations = state[:, 3]
    if trails is not None:
        _draw_trails(screen, w2s, trails, positions)
    if layers.show_sensor:
        _draw_sensors(screen, w2s, positions, orientations, loaded,
                      layers.nest_sensor_px, layers.obj_sensor_px, layers.obj_sensor_fov)

    for i in range(len(state)):
        center = w2s.xy(positions[i, 0], positions[i, 1])
        zones  = layers.zones_deliver if loaded[i] else layers.zones_search
        if zones is not None:
            _draw_zones(screen, center, orientations[i], zones)
        _draw_robot(screen, font, i, center, orientations[i],
                    STATE_RGB[Zone(int(state[i, 7]))], layers.body_px, bool(loaded[i]))


# ── Animación ─────────────────────────────────────────────────────────────────

def _open_output(
    save_path:   str | None,
    screen_size: int,
    interval:    int,
) -> tuple[pygame.Surface, "cv2.VideoWriter | None"]:
    """Inicializa Pygame y crea la superficie de dibujo y, si aplica, el video.

    Args:
        save_path:   Archivo mp4 de salida; None = ventana.
        screen_size: Lado de la ventana / del video (px).
        interval:    Milisegundos entre cuadros (fija los cuadros por segundo del video).

    Returns:
        Tupla (screen, writer): superficie de dibujo y escritor de video
        (None si se muestra en una ventana).
    """
    if not pygame.display.get_init():
        pygame.init()
    pygame.font.init()
    if save_path:
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(1, 1000 // max(1, interval)), (screen_size, screen_size))
        return pygame.Surface((screen_size, screen_size)), writer
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("RAOI — Farming Task")
    return screen, None


def animate_farming(
    report:          np.ndarray,
    objects_report:  np.ndarray,
    carrying_report: np.ndarray,
    env:             dict,
    interval:        int,
    show_zones:      bool,
    show_trail:      bool,
    trail_length:    int,
    show_sensor:     bool,
    save_path:       str | None,
    screen_size:     int,
) -> None:
    """Reproduce la simulación cuadro a cuadro.

    Con save_path=None se muestra en una ventana Pygame a un cuadro cada
    `interval` ms (ESC o cerrar la ventana la detiene). Con save_path se
    renderiza fuera de pantalla, tan rápido como se pueda, y se escribe un
    video mp4 a 1000 / interval cuadros por segundo. El video se cierra y
    Pygame se libera aunque ocurra un error durante el dibujo.

    Args:
        report:          Estado del enjambre, shape (T, N, 8) (ver farming.run()).
        objects_report:  Posiciones de los objetos, shape (T, O, 2).
        carrying_report: Objeto cargado por cada robot (-1 = ninguno), shape (T, N).
        env:             Escenario, con claves:
                           'area_limits', 'plots', 'plot_repulsion',
                           'strip_width', 'strip_gap', 'plot_visual_thickness',
                           'nest_position', 'nest_ri', 'nest_rs',
                           'obj_sensor_range' (alcance desde el centro, m),
                           'obj_sensor_fov' (rad),
                           'nest_icon_radius', 'deposit_radius',
                           'raoi_radii_search', 'raoi_radii_deliver'
                           (radios efectivos de repulsión, orientación y
                           atracción en cada fase, en m).
        interval:        Milisegundos entre cuadros.
        show_zones:      Dibujar las zonas RAOI de cada robot (de búsqueda
                         o de entrega según si carga un objeto).
        show_trail:      Dibujar el rastro de cada robot.
        trail_length:    Pasos del rastro.
        show_sensor:     Dibujar el sensor activo de cada robot: sin carga, el
                         cono del sensor de objetos (env['obj_sensor_range'],
                         env['obj_sensor_fov']); cargado, el círculo del
                         sensor de nest (env['nest_ri']).
        save_path:       Archivo mp4 de salida; None = ventana.
        screen_size:     Lado de la ventana / del video (px).
    """
    iterations, n_robots, _ = report.shape
    n_objects = objects_report.shape[1]
    n_plots   = len(env["plots"])
    nest_pos  = env["nest_position"]
    deposit_r = env["deposit_radius"]

    # Objetos entregados en cada cuadro: los que están dentro del círculo de depósito.
    dxo = objects_report[:, :, 0] - nest_pos[0]
    dyo = objects_report[:, :, 1] - nest_pos[1]
    delivered_per_frame = np.sum((dxo * dxo + dyo * dyo) <= deposit_r * deposit_r, axis=1)

    headless       = bool(save_path)   # sin ventana: solo se escribe el video
    screen, writer = _open_output(save_path, screen_size, interval)
    font_sm    = pygame.font.SysFont("Arial", 11)
    font_title = pygame.font.SysFont("Arial", 14, bold=True)
    clock      = pygame.time.Clock()

    w2s    = WorldToScreen(env["area_limits"], screen_size)
    layers = _RobotLayers.from_env(w2s, env, show_zones, show_sensor)
    trails: list[deque[tuple[int, int]]] = [deque(maxlen=trail_length) for _ in range(n_robots)]
    static_layer = _render_static_layer(w2s, env, screen_size)

    progress = tqdm(total=iterations, desc="Animation ", unit="frame", ncols=72,
                    disable=not headless)
    frame = 0
    try:
        while frame < iterations:
            if not headless and any(
                event.type == pygame.QUIT
                or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
                for event in pygame.event.get()
            ):
                break

            screen.blit(static_layer, (0, 0))
            carried_set = {int(o) for o in carrying_report[frame] if o >= 0}
            _draw_farming_objects(screen, w2s, objects_report[frame], carried_set,
                                  nest_pos, deposit_r)
            _draw_swarm(screen, w2s, font_sm, report[frame], carrying_report[frame] >= 0,
                        trails if show_trail else None, layers)
            _draw_legend(screen, font_sm)
            _draw_farming_hud(screen, font_sm, font_title, frame, iterations,
                              n_robots, n_objects, n_plots, delivered_per_frame[frame])

            if writer is not None:
                frame_rgb = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            else:
                pygame.display.flip()
                clock.tick(1000 // max(1, interval))

            frame += 1
            progress.update(1)
    finally:
        progress.close()
        if writer is not None:
            writer.release()
        pygame.quit()

    logger.info("Animacion terminada (%d cuadros).", frame)
    if writer is not None:
        logger.info("Video guardado: %s", save_path)
