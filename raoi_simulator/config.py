# -*- coding: utf-8 -*-
"""
Configuración del simulador RAOI para la tarea de farming.

Todos los parámetros del modelo, del robot, del escenario y de la
visualización residen aquí. Ningún otro módulo debe contener valores
literales — siempre importar desde este archivo.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math

# ── Reproducibilidad ──────────────────────────────────────────────────────────

SEED: int = 42
"""Semilla global. Cada réplica usa SEED + replica_index para independencia estadística."""

# ── Simulación ────────────────────────────────────────────────────────────────

DT: float = 1.0
"""
Paso de tiempo en segundos.

Los voltajes del modelo dinámico están calibrados para DT=1.0 s.
Reducir DT sin recalibrar los voltajes produce desplazamientos
imperceptibles. Si se requiere un paso más fino, escalar los voltajes
proporcionalmente o recalibrar el modelo.
"""

RK4_SUBSTEPS: int = 10
"""
Subdivisiones internas del integrador RK4.

10 subdivisiones dan precisión equivalente al integrador odeint original.
Reducir a 4–5 para mayor velocidad con menor precisión numérica.
"""

# ── Parámetros RAOI ───────────────────────────────────────────────────────────
#
# Nota: farming usa sus propios pesos (FARMING_RAOI_WEIGHTS, ver mas abajo),
# no RAOI_WEIGHTS. RAOI_RADII se mantiene aqui porque visualization.py lo usa
# para dibujar los radios de zona en modo SHOW_ZONES (solo referencia visual,
# no afecta la fisica — farming pasa sus propios r_r/o_r/a_r a run()).

ROBOT_BODY_RADIUS: float = 0.075
"""Radio físico del robot (m). Ajustable según el diseño del robot. Farming
escala este valor por FARMING_ROBOT_SCALE (ver seccion FARMING)."""

RAOI_RADII: dict = {
    "r_repulsion":   0.075,
    "r_orientation": 1.0,
    "r_attraction":  2.0,
}
"""
Radios de zona RAOI en metros — solo para referencia visual (SHOW_ZONES).

Estos valores se suman a ROBOT_BODY_RADIUS en tiempo de ejecución.
"""

RAOI_FOV: dict = {
    "fov_repulsion":   math.pi,        # ±90°  — frontal
    "fov_orientation": 2 * math.pi,    # 360°  — omnidireccional
    "fov_attraction":  math.pi,        # ±90°  — frontal
    "fov_influence":   math.pi,        # ±90°  — frontal
}
"""
Campos de visión por zona (radianes).

math.pi  → 180° (semicírculo frontal).
2*math.pi → 360° (omnidireccional).
"""

# ── Modelo dinámico — robot diferencial ──────────────────────────────────────
#
# Nota: farming escala mass/inertia/d/wheel_r/wheel_sep por FARMING_ROBOT_SCALE
# al instanciar DynamicsConstants (ver farming.run()). Estos son los valores
# base (robot pequeño) que dynamics.DynamicsConstants usa como default cuando
# no se pasa un override — farming siempre pasa overrides escalados, excepto
# para el motor (MOTOR_Ts/Ks/Kl), que se deja sin escalar a propósito.

ROBOT_MASS: float       = 0.38      # Masa total (kg)
ROBOT_INERTIA: float    = 0.005     # Momento de inercia (kg·m²)
ROBOT_D: float          = 0.02      # Distancia centroide → eje de ruedas (m)
ROBOT_WHEEL_R: float    = 0.03      # Radio de rueda (m)
ROBOT_WHEEL_SEP: float  = 0.05      # Semiseparación entre ruedas (m)

MOTOR_Ts: float = 0.434             # Constante de tiempo del motor (s)
MOTOR_Ks: float = 2.745             # Ganancia de velocidad (rad / s·N·m)
MOTOR_Kl: float = 1460.2705         # Ganancia de corriente (rad / s·V)

VOLTAGE: dict = {
    "repulsion":   2.0,   # ~15 cm/s
    "orientation": 2.7,   # ~20 cm/s
    "attraction":  3.7,   # ~30 cm/s
}
"""
Voltajes de referencia por estado RAOI (V).

La influencia usa interpolación lineal entre V_repulsion y V_attraction
en función de la distancia normalizada a la fuente.
"""

# ── Límites físicos del robot ──────────────────────────────────────────────────
#
# Nota: farming escala V_MAX_LINEAR por FARMING_ROBOT_SCALE (ver farming.run(),
# pasado como override a dynamics.integrate_robot). OMEGA_MAX se deja sin
# escalar a propósito (ver docstring de FARMING_ROBOT_SCALE).

OMEGA_MAX: float    = 10.0   # Velocidad angular máxima (rad/s)
V_MAX_LINEAR: float = 0.5    # Velocidad lineal máxima (m/s), base (sin escalar)

# ── Controlador de giro proporcional ─────────────────────────────────────────

KP_TURN: float = 0.8
"""
Ganancia del controlador de voltaje diferencial.

v_diff = KP_TURN * theta_error  →  voltajes = [v_base + v_diff, v_base - v_diff]

KP_TURN = 0.0 → voltajes simétricos (sin giro activo)
KP_TURN = 0.5 → giro suave
KP_TURN = 2.0 → giro agresivo
"""

# ── Escenario ─────────────────────────────────────────────────────────────────

FARMING_AREA_LIMITS: float = 100.0
"""Lado del área cuadrada de simulación (m)."""

# ── Zona de spawn ─────────────────────────────────────────────────────────────

SPAWN_MIN_SEPARATION: float = 0.3
"""
Separación mínima entre robots en el spawn de farming (m).

En tiempo de ejecución se eleva automáticamente a
FARMING_SPAWN_SPACING_FACTOR*r_repulsion si este valor resulta menor.
"""

# ── Exploración libre ─────────────────────────────────────────────────────────

EXPLORE_FREE_ITERS: int   = 10
"""
Iteraciones consecutivas en estado libre antes de activar la exploración
tipo vuelo de Lévy (ver FARMING_LEVY_* en la sección de farming).

Durante las primeras EXPLORE_FREE_ITERS iteraciones sin nada detectado el
robot mantiene su última dirección activa.
"""

EXPLORE_TURN_NOISE: float = 0.15
"""
Amplitud del giro gaussiano por iteración durante una racha recta del
vuelo de Lévy (se usa escalado x0.2 en farming — ver farming.run()).

0.15 rad ≈ ±8.6°.
"""

DIREXP_RESET_NOISE: float = 0.1
"""
Perturbación gaussiana aplicada a dirExp al entrar en estado libre (rad).

Rompe la inercia de la última dirección activa sin redirigir bruscamente.
"""

# ── Visualización ─────────────────────────────────────────────────────────────

ROBOT_VISUAL_SCALE: float = 1.5
"""Multiplicador del radio visual del robot en la animación Pygame."""

SHOW_ROBOT_IDS: bool  = True
"""Mostrar número de ID sobre cada robot en la animación."""

SHOW_ZONES: bool      = False
"""Mostrar radios de percepción RAOI alrededor de cada robot."""

SHOW_TRAIL: bool      = False
"""Mostrar rastro de trayectoria de los últimos TRAIL_LENGTH pasos."""

TRAIL_LENGTH: int     = 15
"""Número de pasos mostrados en el rastro de trayectoria."""

SCREEN_SIZE: int       = 800
"""Tamaño de la ventana Pygame en píxeles (cuadrada)."""

VIDEO_SAVE_PATH: str   = "simulation.mp4"
"""
Ruta del archivo de video grabado con OpenCV.

Cambiar a None para desactivar la grabación.
Requiere ffmpeg instalado para el codec mp4v.
"""

FARMING_ANIMATION_INTERVAL: int = 20
"""
Milisegundos entre frames en la animación de farming (Pygame).

Farming usa un área de 100×100 m con miles de iteraciones, por lo que se
reproduce a mayor velocidad (20 ms = 50 fps) que un video en tiempo real.
La simulación física no cambia — solo la velocidad de reproducción.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Parámetros exclusivos de la tarea de FARMING
# ══════════════════════════════════════════════════════════════════════════════
# Referencia:
#   Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
#   based on repulsion, attraction and influence. Autonomous Robots.

# ── Robot físico escalado ───────────────────────────────────────────────────────

FARMING_ROBOT_SCALE: float = 6.0
"""
Factor de escala lineal del robot (adimensional).

El robot de farming es más grande que un robot miniatura de laboratorio,
así que sus propiedades físicas se escalan como un sólido uniforme:
dimensiones lineales (ROBOT_D, ROBOT_WHEEL_R, ROBOT_WHEEL_SEP,
ROBOT_BODY_RADIUS) por FARMING_ROBOT_SCALE; masa por
FARMING_ROBOT_SCALE**3; inercia por FARMING_ROBOT_SCALE**5. La velocidad
lineal máxima también escala (V_MAX_LINEAR por FARMING_ROBOT_SCALE, ver
dynamics.integrate_robot) porque una rueda más grande recorre más
distancia por vuelta a la misma velocidad angular del motor.

Los parámetros del motor (MOTOR_Ts/Ks/Kl) y OMEGA_MAX no se escalan: no
hay una regla geométrica que los derive del factor de escala, y
escalarlos sin datos reales del motor sería inventar números. Si el
robot físico usa otro motor, sus constantes se ajustan aparte con datos
reales de ese motor.

dynamics.DynamicsConstants y dynamics.integrate_robot aceptan estos
valores como overrides opcionales (ver sus firmas): si no se pasan,
usan los valores base de ROBOT_*/V_MAX_LINEAR de este archivo.
"""

# ── Comportamiento emergente ─────────────────────────────────────────────────────

FARMING_RAOI_WEIGHTS: dict = {
    "w_r": 0.8,   # Repulsion — dominante, es de seguridad (evita colisiones).
    "w_o": 0.2,   # Orientacion — bajo, para no flockear en bloque.
    "w_a": 0.15,  # Atraccion  — bajo, para no forzar cohesion entre robots.
    "w_I": 0.4,   # Influencia — alto, prioriza reaccionar a lo detectado.
}
"""
Pesos del modelo RAOI para farming.

Con pesos de orientación/atracción altos, el enjambre tiende a flockear
como bloque: cuando ningún robot detecta nada (lo usual, con sensores de
corto alcance sobre una parcela larga), la orientación/atracción entre
vecinos domina y el grupo se mueve junto en vez de dispersarse a cubrir
la franja. Mantener w_o/w_a bajos y w_I alto favorece que cada robot
reaccione más a lo que él mismo detecta (objeto, nest) y menos a lo que
hacen sus vecinos — más parecido a forrajeo individual real, con mejor
cobertura de un área alargada.
No afecta a aggregation/foraging/prey_predator: siguen usando el
RAOI_WEIGHTS global sin cambios.
"""

FARMING_LEVY_MIN_RUN: int = 20
"""Duracion minima (iteraciones) de una racha recta de exploracion tipo vuelo de Levy."""

FARMING_LEVY_MAX_RUN: int = 300
"""Duracion maxima (iteraciones) de una racha recta — evita rachas absurdamente largas."""

FARMING_LEVY_EXPONENT: float = 1.5
"""
Exponente de la distribucion de Pareto que define la duracion de cada racha.

Valores tipicos en la literatura de forrajeo animal (vuelo de Levy) estan
entre 1 y 3; 1.5 da una mezcla de rachas cortas frecuentes y rachas largas
ocasionales (cola pesada), en vez del paso fijo uniforme de un random walk
gaussiano puro. Cubre area dispersa de forma mas eficiente sin dejar de
ser 100% local/descentralizado — solo cambia como cada robot genera su
propio angulo de exploracion cuando no detecta nada.
"""

FARMING_EDGE_FOLLOW_WEIGHT: float = 0.4
"""
Peso del sesgo de "bordeo" al ser repelido por una parcela (0-1).

Cuando la repulsion viene (al menos en parte) de una parcela, se mezcla el
vector de repulsion puro con una componente tangencial (perpendicular a la
repulsion, en la direccion mas alineada con el heading actual del robot),
en vez de solo alejarse en linea recta. 0 = comportamiento anterior (solo
aleja); 1 = solo tangencial (bordea sin alejarse). 0.4 prioriza alejarse
un poco pero favorece recorrer el borde de la parcela en vez de rebotar
y quedar dando vueltas cerca del mismo punto — util porque la parcela es
un obstaculo alargado (75 m), no puntual.
"""

# ── Spawn inicial ────────────────────────────────────────────────────────────

FARMING_SPAWN_LINE_X: float = 0.5
"""
Posicion X de la linea de spawn (m).

Los robots parten alineados en una columna vertical en x=FARMING_SPAWN_LINE_X
(fuera del nest, no dentro), espaciados en Y con separacion
FARMING_SPAWN_SPACING_FACTOR * r_repulsion, todos con la misma orientacion
inicial (theta=0, mirando hacia +X, hacia las parcelas — ver farming.run()).
Reduce el desplazamiento inicial hasta la primera deteccion util, evita el
amontonamiento dentro del area cuadrada del nest cuando individuals es
grande, y elimina la aleatoriedad de la orientacion de arranque.
"""

FARMING_SPAWN_SPACING_FACTOR: float = 3.0
"""
Factor multiplicador de r_repulsion para el espaciado ideal entre robots
en la línea de spawn (m implícitos: spacing = factor * r_repulsion).

Un factor de 3.0 deja bastante aire entre robots vecinos al arrancar —
más que el mínimo seguro de 2*r_repulsion (justo el punto en que las
zonas de repulsión mutua apenas no se tocan) — para que el primer
movimiento de cada uno dependa menos de la repulsión inmediata de su
vecino y más de su propia búsqueda. Si el número de robots no cabe en el
área con este espaciado ideal, run() lo reduce automáticamente hasta el
mínimo seguro (ver farming.run()).
"""

# ── Parcelas ──────────────────────────────────────────────────────────────────

FARMING_PLOT_LENGTH: float = 75.0
"""
Longitud de cada segmento de parcela (m).

El segmento se centra en X dentro del área. Con FARMING_AREA_LIMITS=100:
el segmento va de x=(10-6)/2=2.0 a x=8.0.
Debe ser < AREA_LIMITS para dejar pasillos laterales navegables.
"""

FARMING_PLOT_REPULSION: float = 0.6
"""
Radio de repulsión virtual de las parcelas (m).

Los robots detectan la parcela como fuente repulsiva cuando su distancia
al punto más cercano del segmento es ≤ este valor. Debe ser mayor que
ROBOT_BODY_RADIUS efectivo (0.075 * FARMING_ROBOT_SCALE = 0.45 m) para
preservar un margen de seguridad, y suficientemente menor que la mitad
de FARMING_PLOT_SEPARATION para no cerrar el corredor de tránsito entre
parcelas adyacentes (ver el cálculo de corredor libre en
FARMING_PLOT_SEPARATION).
"""

FARMING_STRIP_WIDTH: float = 1.1
"""
Ancho de la franja donde se colocan objetos a cada lado de la parcela (m).

margen + strip_width debe quedar por debajo de FARMING_PLOT_SEPARATION/2
para que la franja de objetos no invada el territorio de la parcela
vecina. Con FARMING_PLOT_REPULSION=0.6, margen = 0.6+0.1 = 0.7 m, así que
0.7 + 1.1 = 1.8 m frente a un límite de 2.0 m (FARMING_PLOT_SEPARATION/2).
"""

FARMING_PLOT_SEPARATION: float = 4
"""
Distancia centro a centro entre parcelas adyacentes (m).

Con FARMING_PLOT_REPULSION=0.6 y el robot escalado (diámetro ≈0.9 m), el
corredor libre entre dos filas adyacentes es 4.0 - 2*0.6 = 2.8 m — unas
3 veces el diámetro del robot, cómodo para maniobrar.
"""

FARMING_MAX_PLOTS: int = 20
"""
Número máximo de parcelas que caben en el área con los parámetros actuales.

Calculado como: floor((FARMING_AREA_LIMITS - 2*FARMING_PLOT_MARGIN_Y - 2*(STRIP_WIDTH+REPULSION))
                      / SEPARATION) + 1
Se recalcula automáticamente en run() y se usa para validar n_plots del usuario.
"""

FARMING_PLOT_MARGIN_Y: float = 15.0
"""
Margen en Y en los extremos del área (m).

Reserva espacio para el nest (esquina SW) y para maniobra libre de robots
en los bordes superior e inferior sin interferir con las parcelas.
"""

"""
Ancho de la franja de objetos a cada lado de la parcela (m).

Los objetos se distribuyen aleatoriamente en y ∈ [y_plot + REPULSION, y_plot + REPULSION + STRIP_WIDTH]
(franja superior) y simétricamente en la franja inferior.
"""

# ── Nest ──────────────────────────────────────────────────────────────────────

FARMING_NEST_AREA_SIDE: float = 0.2 * FARMING_AREA_LIMITS
"""Lado del área cuadrada del nest (m). Idéntico a foraging."""

FARMING_NEST_POSITION: list = [
    FARMING_NEST_AREA_SIDE / 2,
    FARMING_NEST_AREA_SIDE / 2,
]
"""Posición [x, y] del centro del nest (m). Esquina suroeste del área."""

FARMING_NEST_RI: float = 10.0
"""
Radio sensor del robot hacia el nest (m). Componente r_i (receptor).

El nest SI emite una señal de influencia activa (beacon): el rango efectivo
de detección es FARMING_NEST_RI + FARMING_NEST_RS. Ambos son parámetros de
usuario en run() (nest_ri, nest_rs) — estos son solo los valores por defecto
cuando no se especifican. Con los defaults (10.0 + 2.0 = 12.0 m).
"""

FARMING_NEST_RS: float = 2.0
"""
Radio de emisión de influencia del nest (m). Componente r_s (emisor).

Se suma a FARMING_NEST_RI para dar el rango efectivo de detección. A
diferencia del objeto (que no emite nada — ver FARMING_OBJECT_DETECTION_RADIUS),
el nest sí es una fuente activa de señal, por eso tiene un componente rs
propio que se configura por separado.
"""

FARMING_DEPOSIT_RADIUS: float = 0.5
"""
Radio de depósito efectivo (m).
Debe ser > desplazamiento por iteración (~0.3 m/iter con DT=1.0).
"""

# ── Recolección ───────────────────────────────────────────────────────────────

FARMING_PICK_RADIUS: float = 0.4
"""
Radio de recolección efectivo (m).

Debe ser > desplazamiento por iteración (~0.3 m/iter con DT=1.0) para
garantizar que el robot no pase por encima del objeto sin detectarlo.
"""

FARMING_OBJECT_DETECTION_RADIUS: float = 1.0
"""
Radio de deteccion del robot hacia un objeto (m). Sensor tipo camara.

A diferencia del nest, el objeto NO emite ninguna señal de influencia — es
un elemento pasivo que el robot detecta por proximidad/vision simulada
(un unico componente, sin r_s que sumar). Parametro de usuario en run()
(obj_r) — este es solo el valor por defecto cuando no se especifica.

Con FARMING_PLOT_LENGTH=75 m, un radio de 1 m implica que encontrar un
objeto especifico depende casi enteramente de la exploracion libre y del
barrido colectivo del enjambre sobre la franja, no de que el objeto sea
"visible" desde lejos — es la contrapartida directa de un sensor de corto
alcance realista en vez de un beacon de area.
"""

# ── Límite de iteraciones ──────────────────────────────────────────────────────

FARMING_DEFAULT_MAX_ITER: int = 7200
"""
Número de iteraciones por defecto de una simulación de farming.

Se usa siempre que run() no reciba time_limit explícito. El bucle puede
terminar antes si todos los objetos son entregados (ver run()).
"""