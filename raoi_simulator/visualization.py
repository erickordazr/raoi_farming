# -*- coding: utf-8 -*-
"""
Módulo de visualización de la tarea de farming.

Renderiza robots, objetos, parcelas y nest cuadro a cuadro con Pygame,
con grabación opcional de video (.mp4) usando OpenCV.

Capas opcionales desde config.py:
  SHOW_ZONES  → sectores de percepción RAOI
  SHOW_TRAIL  → rastro de los últimos TRAIL_LENGTH pasos

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math
import os
import numpy as np
import pygame
import cv2
from collections import deque
from typing import Optional

from . import config


# ── Paleta ────────────────────────────────────────────────────────────────────

BG_COLOR        = (255, 255, 255)
BORDER_COLOR    = ( 40,  40,  40)
GRID_COLOR      = (220, 220, 220)
TEXT_COLOR      = ( 30,  30,  30)
TRAIL_ALPHA     = 120   # 0-255

PLOT_COLOR      = ( 60, 120,  50)   # verde oscuro — segmento de parcela
PLOT_FILL_COLOR = (200, 230, 190)   # verde claro  — franja de objetos
FRUIT_COLOR     = (220,  80,  30)   # naranja-rojo — objeto disponible
FRUIT_CARRIED   = (255, 140,  60)   # naranja claro — borde de objeto cargado

STATE_RGB = {
    0: (160, 160, 160),   # Sin vecinos   — gris
    1: (220,  50,  60),   # Repulsión     — rojo
    2: ( 60, 120, 170),   # Atracción     — azul
    3: ( 38, 160, 140),   # Orientación   — verde
    4: (220, 185,  80),   # Influencia    — dorado
}
STATE_LABELS = {
    0: "Free exploration",
    1: "Repulsion",
    2: "Attraction",
    3: "Orientation",
    4: "Influence",
}
ZONE_RGBA = {
    "repulsion":   (220,  50,  60, 35),
    "orientation": ( 38, 160, 140, 20),
    "attraction":  ( 60, 120, 170, 15),
}


def _rotate_points(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rota un array de puntos (N,2) alrededor del origen."""
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def _robot_polygons(cx: float, cy: float, theta: float, r: float) -> dict:
    """
    Calcula los vértices del icono de robot diferencial en píxeles.

    Cuerpo circular con ruedas gruesas y nariz frontal triangular.
    La nariz actúa como indicador de dirección integrado al cuerpo.
    La flecha sale desde la punta de la nariz hacia adelante.

    Args:
        cx, cy: Centro en píxeles.
        theta:  Orientación en radianes (coordenadas de simulación).
        r:      Radio visual del cuerpo en píxeles.

    Returns:
        Dict con:
          'body_center'  — (cx, cy, r) para pygame.draw.circle
          'nose'         — triángulo de nariz frontal (polígono)
          'wheel_l'      — rectángulo rueda izquierda
          'wheel_r'      — rectángulo rueda derecha
          'arrow'        — ((x0,y0), (x1,y1)) flecha desde nariz
    """
    # ── Nariz frontal: triángulo que sale del círculo hacia adelante ──────────
    nose_pts = np.array([
        [ r * 1.45,  0.0      ],   # punta
        [ r * 0.80,  r * 0.38],   # base derecha  (tangente al círculo)
        [ r * 0.80, -r * 0.38],   # base izquierda
    ])

    # ── Ruedas: gruesas y prominentes ─────────────────────────────────────────
    wl  = r * 1.10   # largo (cubre longitud del cuerpo)
    wh  = r * 0.45   # ancho — prominente y visible
    wy  = r * 1.00   # distancia lateral al borde interno

    wl_pts = np.array([          # rueda izquierda (+y local)
        [-wl/2,  wy      ],
        [ wl/2,  wy      ],
        [ wl/2,  wy + wh ],
        [-wl/2,  wy + wh ],
    ])
    wr_pts = np.array([          # rueda derecha (-y local)
        [-wl/2, -wy - wh],
        [ wl/2, -wy - wh],
        [ wl/2, -wy     ],
        [-wl/2, -wy     ],
    ])

    # ── Flecha: desde punta de nariz hacia adelante ───────────────────────────
    arrow_start = np.array([[r * 1.45, 0.0]])
    arrow_end   = np.array([[r * 2.10, 0.0]])

    # Rotar todo con -theta (pygame Y invertido)
    angle  = -theta
    center = np.array([cx, cy])

    nose_r  = _rotate_points(nose_pts,    angle) + center
    wl_r    = _rotate_points(wl_pts,      angle) + center
    wr_r    = _rotate_points(wr_pts,      angle) + center
    arr_s   = (_rotate_points(arrow_start, angle) + center)[0]
    arr_e   = (_rotate_points(arrow_end,   angle) + center)[0]

    return {
        "body_center": (cx, cy, r),
        "nose":        nose_r.tolist(),
        "wheel_l":     wl_r.tolist(),
        "wheel_r":     wr_r.tolist(),
        "arrow":       (arr_s, arr_e),
    }


# ── Conversión mundo → pantalla ───────────────────────────────────────────────

class WorldToScreen:
    """
    Convierte coordenadas de simulación (metros) a píxeles de pantalla.
    Pygame tiene Y=0 arriba, la simulación tiene Y=0 abajo.
    """
    def __init__(self, area_m: float, screen_px: int, margin_px: int = 40):
        self.area_m    = area_m
        self.screen_px = screen_px
        self.margin    = margin_px
        self.drawable  = screen_px - 2 * margin_px
        self.scale     = self.drawable / area_m   # px/m

    def xy(self, xm: float, ym: float) -> tuple[int, int]:
        """Metro → píxel (con Y invertido)."""
        px = int(self.margin + xm * self.scale)
        py = int(self.margin + (self.area_m - ym) * self.scale)
        return px, py

    def r(self, rm: float) -> int:
        """Radio en metros → radio en píxeles."""
        return max(1, int(rm * self.scale))


def _draw_farming_environment(
    surf: pygame.Surface,
    w2s: WorldToScreen,
    env: dict,
) -> None:
    """
    Dibuja el entorno de farming: nest, parcelas y franjas de objetos.

    Args:
        surf: Superficie pygame.
        w2s:  Conversor mundo→pantalla.
        env:  Dict con 'nest_position', 'nest_radius', 'nest_area_side',
              'plots', 'plot_repulsion', 'strip_width'.
    """
    # ── Nest (idéntico al de foraging) ────────────────────────────────────────
    nest_pos   = env["nest_position"]
    nest_area  = env["nest_area_side"]
    nest_rad   = env.get("nest_radius", 4.0)

    # Radio de influencia del nest
    nr_px = w2s.r(nest_rad)
    nc_px = w2s.xy(nest_pos[0], nest_pos[1])
    nest_surf = pygame.Surface((nr_px * 2 + 2, nr_px * 2 + 2), pygame.SRCALPHA)
    pygame.draw.circle(nest_surf, (100, 180, 100, 35), (nr_px + 1, nr_px + 1), nr_px)
    surf.blit(nest_surf, (nc_px[0] - nr_px - 1, nc_px[1] - nr_px - 1))

    # Área del nest
    half = nest_area / 2
    n_lo = w2s.xy(nest_pos[0] - half, nest_pos[1] + half)
    n_hi = w2s.xy(nest_pos[0] + half, nest_pos[1] - half)
    n_w  = n_hi[0] - n_lo[0]
    n_h  = n_hi[1] - n_lo[1]
    nest_s = pygame.Surface((max(1, n_w), max(1, n_h)), pygame.SRCALPHA)
    nest_s.fill((80, 160, 80, 90))
    surf.blit(nest_s, n_lo)
    pygame.draw.rect(surf, (50, 130, 50), (n_lo[0], n_lo[1], max(1, n_w), max(1, n_h)), 2)

    # Etiqueta NEST
    try:
        font_nest = pygame.font.SysFont("Arial", 11, bold=True)
        lbl = font_nest.render("NEST", True, (30, 100, 30))
        surf.blit(lbl, (n_lo[0] + 2, n_lo[1] + 2))
    except Exception:
        pass

    # ── Franjas de objetos (fondo semitransparente) ───────────────────────────
    strip_width    = env.get("strip_width", 1.2)
    plot_repulsion = env.get("plot_repulsion", 0.4)
    margin         = plot_repulsion + 0.05

    for plot in env.get("plots", []):
        x0, x1, yp = plot["x0"], plot["x1"], plot["y"]

        # Franja superior
        yf_lo = yp + margin;       yf_hi = yp + margin + strip_width
        px_lo = w2s.xy(x0, yf_hi); px_hi = w2s.xy(x1, yf_lo)
        fw    = max(1, px_hi[0] - px_lo[0])
        fh    = max(1, px_hi[1] - px_lo[1])
        fs    = pygame.Surface((fw, fh), pygame.SRCALPHA)
        fs.fill((*PLOT_FILL_COLOR, 60))
        surf.blit(fs, px_lo)

        # Franja inferior
        yf_lo = yp - margin - strip_width; yf_hi = yp - margin
        px_lo = w2s.xy(x0, yf_hi);         px_hi = w2s.xy(x1, yf_lo)
        fw    = max(1, px_hi[0] - px_lo[0])
        fh    = max(1, px_hi[1] - px_lo[1])
        fs    = pygame.Surface((fw, fh), pygame.SRCALPHA)
        fs.fill((*PLOT_FILL_COLOR, 60))
        surf.blit(fs, px_lo)

    # ── Segmentos de parcela ──────────────────────────────────────────────────
    plot_rep_px = w2s.r(plot_repulsion)

    for plot in env.get("plots", []):
        x0, x1, yp = plot["x0"], plot["x1"], plot["y"]
        p0 = w2s.xy(x0, yp)
        p1 = w2s.xy(x1, yp)

        # Zona de repulsión semitransparente
        rep_w = p1[0] - p0[0]
        rep_h = max(1, plot_rep_px * 2)
        rep_s = pygame.Surface((max(1, rep_w), rep_h), pygame.SRCALPHA)
        rep_s.fill((200, 100, 50, 30))
        surf.blit(rep_s, (p0[0], p0[1] - plot_rep_px))

        # Línea principal de la parcela
        pygame.draw.line(surf, PLOT_COLOR, p0, p1, 4)

        # Marcas de extremo
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
    nest_area: float,
) -> None:
    """
    Dibuja los frutos disponibles. Los cargados se omiten (se dibujan en el robot).
    Los entregados (dentro del área del nest) se dibujan en verde.

    Args:
        surf:          Superficie pygame.
        w2s:           Conversor mundo→pantalla.
        obj_positions: Posiciones actuales de objetos, shape (O, 2).
        carried_set:   Índices de objetos actualmente cargados.
        nest_pos:      Posición [x, y] del nest.
        nest_area:     Lado del área del nest (m).
    """
    half    = nest_area / 2
    obj_r   = max(5, w2s.r(0.15))

    for o, pos in enumerate(obj_positions):
        if o in carried_set:
            continue

        cx, cy   = w2s.xy(pos[0], pos[1])
        in_nest  = (abs(pos[0] - nest_pos[0]) <= half
                    and abs(pos[1] - nest_pos[1]) <= half)

        if in_nest:
            # Fruto entregado — verde pequeño
            pygame.draw.circle(surf, (60, 160, 60),  (cx, cy), max(3, obj_r - 2))
            pygame.draw.circle(surf, (30, 100, 30),  (cx, cy), max(3, obj_r - 2), 1)
        else:
            # Fruto disponible — naranja
            pygame.draw.circle(surf, (90, 45, 0),    (cx + 2, cy + 2), obj_r)  # sombra
            pygame.draw.circle(surf, FRUIT_COLOR,    (cx, cy), obj_r)
            pygame.draw.circle(surf, (255, 200, 100),(cx, cy), obj_r, 1)


def _draw_farming_hud(
    surf: pygame.Surface,
    font_sm,
    font_lg,
    font_title,
    frame: int,
    iterations: int,
    n_robots: int,
    n_objects: int,
    n_plots: int,
    delivered: int,
) -> None:
    """
    Dibuja el HUD de farming: frame, progreso de entrega y parámetros.

    Args:
        surf:       Superficie pygame.
        font_*:     Fuentes pygame.
        frame:      Frame actual.
        iterations: Total de frames.
        n_robots:   Número de robots.
        n_objects:  Número de objetos.
        n_plots:    Número de parcelas.
        delivered:  Objetos entregados en este frame.
    """
    W, H = surf.get_size()

    # Título
    title = font_title.render("RAOI — Farming Task", True, TEXT_COLOR)
    surf.blit(title, (W // 2 - title.get_width() // 2, 6))

    # Frame
    frame_txt = font_sm.render(f"Frame {frame + 1} / {iterations}", True, TEXT_COLOR)
    surf.blit(frame_txt, (W - frame_txt.get_width() - 10, 6))

    # Barra de progreso de entrega
    bar_w = 160; bar_h = 14
    bar_x = W - bar_w - 10
    bar_y = 26
    frac  = delivered / max(n_objects, 1)
    pygame.draw.rect(surf, (220, 220, 220), (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(surf, FRUIT_COLOR,    (bar_x, bar_y, int(bar_w * frac), bar_h))
    pygame.draw.rect(surf, (80, 80, 80),   (bar_x, bar_y, bar_w, bar_h), 1)
    prog_txt = font_sm.render(f"{delivered}/{n_objects} fruits", True, TEXT_COLOR)
    surf.blit(prog_txt, (bar_x - prog_txt.get_width() - 6,
                          bar_y + (bar_h - prog_txt.get_height()) // 2))

    # Info inferior
    info = font_sm.render(
        f"Robots: {n_robots}   Plots: {n_plots}   Objects: {n_objects}",
        True, TEXT_COLOR,
    )
    surf.blit(info, (10, H - info.get_height() - 6))


def animate_farming(
    report:          np.ndarray,
    objects_report:  np.ndarray,
    carrying_report: np.ndarray,
    env:             dict,
    interval:        int   = 100,
    show_zones:      bool  = False,
    show_trail:      bool  = False,
    trail_length:    int   = 15,
    save_path:       Optional[str] = None,
    screen_size:     int   = 800,
) -> None:
    """
    Reproduce la animación Pygame de la tarea de farming.

    Renderiza robots, frutos, parcelas y nest cuadro a cuadro.
    Si save_path no es None, escribe también un video mp4 con OpenCV.

    Args:
        report:          Estado del enjambre, shape (T, N, 8).
        objects_report:  Posiciones de objetos, shape (T, O, 2).
        carrying_report: Índice del objeto cargado por cada robot, shape (T, N).
        env:             Dict del escenario con claves:
                            'area_limits', 'nest_position', 'nest_radius',
                            'nest_area_side', 'plots', 'plot_repulsion', 'strip_width'.
        interval:        Milisegundos entre frames.
        show_zones:      Mostrar radios RAOI alrededor de cada robot.
        show_trail:      Mostrar rastro de trayectoria.
        trail_length:    Número de pasos en el rastro.
        save_path:       Ruta del video de salida. None → sin grabación.
        screen_size:     Tamaño de la ventana en píxeles.
    """
    iterations, n_robots, _ = report.shape
    n_objects = objects_report.shape[1]
    n_plots   = len(env.get("plots", []))
    nest_pos  = env["nest_position"]
    nest_area = env["nest_area_side"]

    # Conteo de entregas por frame: objeto entregado = dentro del nest area
    half = nest_area / 2
    delivered_per_frame = np.array([
        int(np.sum(
            (np.abs(objects_report[t, :, 0] - nest_pos[0]) <= half) &
            (np.abs(objects_report[t, :, 1] - nest_pos[1]) <= half)
        ))
        for t in range(iterations)
    ])

    # ── Inicializar Pygame ────────────────────────────────────────────────────
    headless = (save_path is not None and not pygame.display.get_init())
    if not pygame.display.get_init():
        pygame.init()

    if save_path:
        screen = pygame.Surface((screen_size, screen_size))
    else:
        screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption("RAOI — Farming Task")

    pygame.font.init()
    font_sm    = pygame.font.SysFont("Arial", 11)
    font_lg    = pygame.font.SysFont("Arial", 13, bold=True)
    font_title = pygame.font.SysFont("Arial", 14, bold=True)
    clock      = pygame.time.Clock()

    w2s      = WorldToScreen(env["area_limits"], screen_size)
    body_px  = max(6, w2s.r(config.ROBOT_BODY_RADIUS * config.ROBOT_VISUAL_SCALE))
    r_rep_px = w2s.r(config.RAOI_RADII["r_repulsion"])
    r_ori_px = w2s.r(config.RAOI_RADII["r_orientation"])
    r_att_px = w2s.r(config.RAOI_RADII["r_attraction"])
    trails   = [deque(maxlen=trail_length) for _ in range(n_robots)]

    # ── OpenCV writer ─────────────────────────────────────────────────────────
    writer = None
    if save_path:
        fps    = max(1, 1000 // max(1, interval))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (screen_size, screen_size))

    frame = 0
    running = True

    while running and frame < iterations:
        if not headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                    break

        # ── Fondo y grid ──────────────────────────────────────────────────────
        screen.fill(BG_COLOR)
        step_g = 1.0
        v = 0.0
        while v <= env["area_limits"]:
            x0, y0 = w2s.xy(v, 0);           x1, y1 = w2s.xy(v, env["area_limits"])
            pygame.draw.line(screen, GRID_COLOR, (x0, y0), (x1, y1), 1)
            x0, y0 = w2s.xy(0, v);           x1, y1 = w2s.xy(env["area_limits"], v)
            pygame.draw.line(screen, GRID_COLOR, (x0, y0), (x1, y1), 1)
            v += step_g

        # Borde del área
        bx0, by0 = w2s.xy(0, env["area_limits"])
        bx1, by1 = w2s.xy(env["area_limits"], 0)
        pygame.draw.rect(screen, BORDER_COLOR, (bx0, by0, bx1 - bx0, by1 - by0), 2)

        # Entorno farming (nest + parcelas + franjas)
        _draw_farming_environment(screen, w2s, env)

        # Frutos: omitir cargados (se dibujan en la nariz del robot)
        carried_set = {
            int(carrying_report[frame, i])
            for i in range(n_robots)
            if carrying_report[frame, i] >= 0
        }
        _draw_farming_objects(
            screen, w2s,
            objects_report[frame],
            carried_set,
            nest_pos, nest_area,
        )

        # ── Robots ────────────────────────────────────────────────────────────
        positions    = report[frame, :, :2]
        orientations = report[frame, :, 3]
        states       = report[frame, :, 7]

        for i in range(n_robots):
            xm, ym = positions[i]
            theta  = orientations[i]
            state  = int(states[i])
            color  = STATE_RGB.get(state, (128, 128, 128))
            cx, cy = w2s.xy(xm, ym)

            trails[i].append((cx, cy))

            if show_zones:
                for (r_px, rgba, fov_key) in [
                    (r_rep_px, ZONE_RGBA["repulsion"],   "fov_repulsion"),
                    (r_ori_px, ZONE_RGBA["orientation"], "fov_orientation"),
                    (r_att_px, ZONE_RGBA["attraction"],  "fov_attraction"),
                ]:
                    fov_v  = config.RAOI_FOV[fov_key]
                    zone_s = pygame.Surface((r_px * 2 + 2, r_px * 2 + 2), pygame.SRCALPHA)
                    if fov_v >= 2 * math.pi - 0.01:
                        pygame.draw.circle(zone_s, rgba, (r_px + 1, r_px + 1), r_px)
                    else:
                        start_a = -theta - fov_v / 2
                        pts = [(r_px + 1, r_px + 1)]
                        steps = max(20, int(math.degrees(fov_v)))
                        for k in range(steps + 1):
                            a = start_a + fov_v * k / steps
                            pts.append((r_px + 1 + r_px * math.cos(a),
                                        r_px + 1 + r_px * math.sin(a)))
                        if len(pts) > 2:
                            pygame.draw.polygon(zone_s, rgba, pts)
                    screen.blit(zone_s, (cx - r_px - 1, cy - r_px - 1))

            if show_trail and len(trails[i]) >= 2:
                pts_t = list(trails[i])
                n_pts = len(pts_t)
                for k in range(n_pts - 1):
                    alpha = int((k + 1) / n_pts * TRAIL_ALPHA)
                    tr_s  = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                    pygame.draw.line(tr_s, (130, 130, 130, alpha),
                                     pts_t[k], pts_t[k + 1], 2)
                    screen.blit(tr_s, (0, 0))

            polys  = _robot_polygons(cx, cy, theta, body_px)
            to_int = lambda pts: [(int(x), int(y)) for x, y in pts]
            bx, by, br = polys["body_center"]

            pygame.draw.circle(screen, (185, 185, 185), (bx + 2, by + 2), br)

            for wkey in ("wheel_l", "wheel_r"):
                wpts = polys[wkey]
                pygame.draw.polygon(screen, (35, 35, 35), to_int(wpts))
                pygame.draw.polygon(screen, (60, 60, 60), to_int(wpts), 1)

            pygame.draw.circle(screen, color, (bx, by), br)
            pygame.draw.circle(screen, (25, 25, 25), (bx, by), br, 1)

            r_c, g_c, b_c = color
            nose_color = (min(255, r_c + 45), min(255, g_c + 45), min(255, b_c + 45))
            pygame.draw.polygon(screen, nose_color, to_int(polys["nose"]))
            pygame.draw.polygon(screen, (25, 25, 25), to_int(polys["nose"]), 1)

            hub_r = max(2, br // 5)
            pygame.draw.circle(screen, (245, 245, 245), (bx, by), hub_r)
            pygame.draw.circle(screen, (40, 40, 40), (bx, by), hub_r, 1)

            arr_s, arr_e = polys["arrow"]
            pygame.draw.line(screen, (15, 15, 15),
                             (int(arr_s[0]), int(arr_s[1])),
                             (int(arr_e[0]), int(arr_e[1])), 2)
            tip_x, tip_y = int(arr_e[0]), int(arr_e[1])
            tip_ang  = -theta
            tip_size = max(5, br // 2)
            tip_pts  = [
                (tip_x, tip_y),
                (int(tip_x - tip_size * math.cos(tip_ang - 0.42)),
                 int(tip_y - tip_size * math.sin(tip_ang - 0.42))),
                (int(tip_x - tip_size * math.cos(tip_ang + 0.42)),
                 int(tip_y - tip_size * math.sin(tip_ang + 0.42))),
            ]
            pygame.draw.polygon(screen, (15, 15, 15), tip_pts)

            # Fruto en la nariz: visual de transporte
            if carrying_report[frame, i] >= 0:
                nose_x = int(bx + body_px * 1.45 * math.cos(-theta))
                nose_y = int(by + body_px * 1.45 * math.sin(-theta))
                obj_r  = max(5, body_px // 2)
                pygame.draw.circle(screen, (120, 50, 0),  (nose_x + 2, nose_y + 2), obj_r)
                pygame.draw.circle(screen, FRUIT_COLOR,   (nose_x, nose_y), obj_r)
                pygame.draw.circle(screen, FRUIT_CARRIED, (nose_x, nose_y), obj_r, 2)

            if config.SHOW_ROBOT_IDS:
                id_surf = font_sm.render(str(i), True, (20, 20, 20))
                id_x    = cx - id_surf.get_width() // 2
                id_y    = cy - br - id_surf.get_height() - 1
                bg_w    = id_surf.get_width() + 4
                bg_h    = id_surf.get_height() + 2
                bg_     = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
                bg_.fill((255, 255, 255, 160))
                screen.blit(bg_,     (id_x - 2, id_y - 1))
                screen.blit(id_surf, (id_x, id_y))

        # ── Leyenda ───────────────────────────────────────────────────────────
        leg_x, leg_y = 10, 44
        leg_surf = pygame.Surface((200, len(STATE_RGB) * 22 + 10), pygame.SRCALPHA)
        leg_surf.fill((255, 255, 255, 200))
        screen.blit(leg_surf, (leg_x - 4, leg_y - 4))
        for state_id, label in STATE_LABELS.items():
            rgb = STATE_RGB[state_id]
            pygame.draw.circle(screen, rgb, (leg_x + 8, leg_y + 8), 7)
            pygame.draw.circle(screen, (30, 30, 30), (leg_x + 8, leg_y + 8), 7, 1)
            txt = font_sm.render(label, True, TEXT_COLOR)
            screen.blit(txt, (leg_x + 20, leg_y + 1))
            leg_y += 22

        # ── HUD ───────────────────────────────────────────────────────────────
        _draw_farming_hud(
            screen, font_sm, font_lg, font_title,
            frame, iterations, n_robots, n_objects, n_plots,
            delivered_per_frame[frame],
        )

        if not headless:
            pygame.display.flip()

        if writer is not None:
            px_array  = pygame.surfarray.array3d(screen)
            frame_bgr = cv2.cvtColor(
                np.transpose(px_array, (1, 0, 2)), cv2.COLOR_RGB2BGR
            )
            writer.write(frame_bgr)

        clock.tick(1000 // max(1, interval))
        frame += 1

        if frame % max(1, iterations // 10) == 0:
            pct = int(frame / iterations * 100)
            print(f"  Animating... {pct}%", end="\r")

    print(f"\nAnimation complete ({frame} frames rendered).")

    if writer is not None:
        writer.release()
        print(f"Video saved: {save_path}")

    pygame.quit()