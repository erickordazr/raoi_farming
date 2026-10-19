"""Configuración del simulador RAOI para la tarea de farming.

Todos los parámetros del modelo, del robot, del escenario y de la
visualización residen aquí. Los demás módulos no contienen valores
numéricos de ajuste: siempre los leen de este archivo.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math

# ══════════════════════════════════════════════════════════════════════════════
# Simulación
# ══════════════════════════════════════════════════════════════════════════════

SEED: int = 42
"""Semilla aleatoria por defecto. Cada réplica estadística usa SEED + índice de réplica."""

DT: float = 1.0
"""Paso de tiempo de la simulación (s). Una iteración = DT segundos."""

RK4_SUBSTEPS: int = 10
"""Subpasos del integrador RK4 dentro de cada DT (más subpasos = más precisión, más costo)."""

FARMING_DEFAULT_MAX_ITER: int = 7200
"""
Iteraciones máximas de una simulación cuando run() no recibe time_limit.
La simulación termina antes si se entregan todos los objetos.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Valores por defecto del menú interactivo
# ══════════════════════════════════════════════════════════════════════════════
# Se usan al dejar en blanco la respuesta en main.py / single_run() /
# statistical_run(); run() siempre recibe estos valores de forma explícita.

FARMING_DEFAULT_N_OBJECTS: int = 250
"""Número de objetos por defecto."""

FARMING_DEFAULT_INDIVIDUALS: int = 25
"""Número de robots por defecto."""

FARMING_DEFAULT_R_REPULSION: float = 0.25
"""Radio de repulsión por defecto en la fase de búsqueda (m, desde el borde del robot)."""

FARMING_DEFAULT_R_ORIENTATION: float = 1.0
"""Radio de orientación por defecto en la fase de búsqueda (m, desde el borde del robot)."""

FARMING_DEFAULT_R_ATTRACTION: float = 2.0
"""Radio de atracción por defecto en la fase de búsqueda (m, desde el borde del robot)."""

FARMING_DEFAULT_ANIMATION: bool = True
"""Respuesta por defecto a la pregunta de animación de la simulación individual."""

# ══════════════════════════════════════════════════════════════════════════════
# Resultados
# ══════════════════════════════════════════════════════════════════════════════

RESULTS_DIR: str = "results"
"""
Carpeta donde single_run() y statistical_run() guardan sus resultados. Cada
corrida crea dentro una subcarpeta con fecha y hora (por ejemplo
results/2026-10-19_14-30-05_single) con los reportes .npy, el video y
REPORT_FILENAME. La carpeta está excluida de git (.gitignore).
"""

REPORT_FILENAME: str = "report.txt"
"""Nombre del reporte de texto con los parámetros y las funciones objetivo f1–f4."""

# ══════════════════════════════════════════════════════════════════════════════
# Robot base (dimensiones de laboratorio, Ordaz-Rivas et al. 2018)
# ══════════════════════════════════════════════════════════════════════════════
#
# Farming escala estas magnitudes con FARMING_ROBOT_SCALE (ver más abajo).

ROBOT_BODY_RADIUS: float = 0.075    # Radio del cuerpo (m)
ROBOT_MASS: float        = 0.38     # Masa total (kg)
ROBOT_INERTIA: float     = 0.005    # Momento de inercia (kg·m²)
ROBOT_D: float           = 0.02     # Distancia del centroide al eje de ruedas (m)
ROBOT_WHEEL_R: float     = 0.03     # Radio de rueda (m)
ROBOT_WHEEL_SEP: float   = 0.05     # Semiseparación entre ruedas (m)

MOTOR_Ts: float = 0.434             # Constante de tiempo del motor (s)
MOTOR_Ks: float = 2.745             # Ganancia de velocidad (rad / s·N·m)
MOTOR_Kl: float = 1460.2705         # Ganancia de corriente (rad / s·V)

OMEGA_MAX: float = 10.0
"""Velocidad angular máxima del robot (rad/s)."""

WALL_CONTACT_GAP: float = 0.01
"""
Holgura (m) con la que el robot queda dentro del margen de pared tras
intentar salir del área (ver dynamics.resolve_walls).
"""

WALL_BOUNCE_HEADING_NOISE_STD: float = 0.2
"""
Desviación estándar (rad) del ruido sumado al rumbo cuando un robot que
apunta hacia la pared se redirige al centro del área (rompe simetrías).
"""

# ══════════════════════════════════════════════════════════════════════════════
# Percepción RAOI
# ══════════════════════════════════════════════════════════════════════════════

RAOI_FOV: dict = {
    "fov_repulsion":   math.pi,        # 180° frontal
    "fov_orientation": 2 * math.pi,    # 360°
    "fov_attraction":  math.pi,        # 180° frontal
}
"""
Campo de visión (rad) de cada zona RAOI entre robots. Un vecino fuera del
campo de visión de una zona no cuenta para esa zona. Los sensores de nest y
de objetos tienen su propio campo de visión (FARMING_FOV_NEST_SENSOR y
FARMING_FOV_OBJECT_SENSOR).
"""

# ══════════════════════════════════════════════════════════════════════════════
# Visualización
# ══════════════════════════════════════════════════════════════════════════════

ROBOT_VISUAL_SCALE: float = 1.5
"""Multiplicador del radio con que se dibuja el robot en la animación."""

SHOW_ROBOT_IDS: bool = True
"""Mostrar el índice de cada robot sobre su icono."""

SHOW_ZONES: bool = False
"""Dibujar las zonas RAOI (repulsión, orientación, atracción) de cada robot."""

SHOW_TRAIL: bool = False
"""Dibujar el rastro de los últimos TRAIL_LENGTH pasos de cada robot."""

TRAIL_LENGTH: int = 15
"""Número de pasos del rastro de trayectoria."""

SHOW_SENSOR_RADIUS: bool = True
"""
Dibujar de forma tenue el sensor activo de cada robot: sin carga, el cono
del sensor de objetos (FARMING_OBJECT_SENSOR_RADIUS, FARMING_FOV_OBJECT_SENSOR);
cargado, el círculo del sensor de nest (FARMING_NEST_RI).
"""

SCREEN_SIZE: int = 800
"""Lado de la ventana / del video (píxeles)."""

SAVE_VIDEO: bool = True
"""
True: la animación se renderiza fuera de pantalla y se graba en un video mp4
(OpenCV, códec mp4v) dentro de la carpeta de la corrida. False: se muestra
en una ventana Pygame y no se graba.
"""

VIDEO_FILENAME: str = "simulation.mp4"
"""Nombre del video de la animación dentro de la carpeta de la corrida."""

FARMING_ANIMATION_INTERVAL: int = 20
"""Milisegundos entre cuadros (20 ms = 50 cuadros por segundo)."""

# ══════════════════════════════════════════════════════════════════════════════
# FARMING — robot escalado y velocidades
# ══════════════════════════════════════════════════════════════════════════════
# Referencia:
#   Ordaz-Rivas et al. (2021). Autonomous foraging with a pack of robots
#   based on repulsion, attraction and influence. Autonomous Robots.

FARMING_ROBOT_SCALE: float = 6.0
"""
Factor de escala lineal del robot de farming respecto al robot base.

Se escala como un sólido uniforme: dimensiones × escala (cuerpo de
0.9 m de diámetro), masa × escala³ (~82 kg), inercia × escala⁵.

El motor se escala por similitud dinámica: MOTOR_Kl / escala⁵ (MOTOR_Ts y
MOTOR_Ks no cambian). En el modelo del actuador la constante de tiempo de
traslación es m·Kl·r²/2 + Ts y la de rotación (I + m·d²)·Kl·r²/(2·R²) + Ts;
con este escalado ambas quedan iguales a las del robot base (~0.68 s): el
robot grande responde tan rápido como el robot base, como un vehículo con
un motor dimensionado para su tamaño. La velocidad de régimen para un
voltaje u es v = Ks·r·u, independiente de Kl. OMEGA_MAX no se escala.
"""

FARMING_ZONE_SPEEDS: dict = {
    "repulsion":   0.4,   # m/s — maniobra de evasión cerca de robots u obstáculos
    "orientation": 0.7,   # m/s — avance de trabajo con vecinos y exploración libre
    "attraction":  1.0,   # m/s — tránsito
}
"""
Velocidad de avance objetivo por zona RAOI (m/s).

farming.run() convierte cada velocidad en el voltaje que la produce en
régimen permanente, u = v / (MOTOR_Ks · r), con r el radio de rueda
escalado. La influencia interpola entre la velocidad de repulsión (encima
del objetivo) y la de atracción (en el borde de detección): el robot
desacelera al acercarse al objeto o al nest. El regreso y la búsqueda
guiados por la brújula usan la velocidad de orientación.

Criterio: plataformas agrícolas y de campo de tamaño similar —
Clearpath Husky A300 (0.99×0.70 m, 80 kg, máx. 2.0 m/s), Saga Thorvald
(180 kg, máx. 1.5 m/s), Naïo Oz (1.30×0.47 m, 150 kg, ~0.5 m/s entre
surcos). La repulsión es 40 % de la atracción, la proporción de
Ordaz-Rivas et al. (2021), Tabla 2.
"""

FARMING_MAX_SPEED: float = 1.0
"""
Velocidad lineal máxima (m/s): tope de seguridad igual a la velocidad de
tránsito. Solo corta excursiones por el ruido de actuador.
"""

# ══════════════════════════════════════════════════════════════════════════════
# FARMING — comportamiento RAOI y exploración
# ══════════════════════════════════════════════════════════════════════════════

FARMING_RAOI_WEIGHTS: dict = {
    "w_r": 0.8,   # Repulsión: dominante, evita colisiones
    "w_o": 0.2,   # Orientación: baja, para no moverse en bloque
    "w_a": 0.15,  # Atracción: baja, para no forzar cohesión
    "w_I": 0.4,   # Influencia: alta, prioriza lo que el propio robot detecta
}
"""
Pesos RAOI de farming. Con orientación/atracción bajas y influencia alta
cada robot reacciona más a lo que detecta (objeto, nest) que a lo que
hacen sus vecinos, y el enjambre se dispersa para cubrir las franjas en
vez de desplazarse como bloque.
"""

FARMING_FOV_NEST_SENSOR: float = 2 * math.pi
"""
Campo de visión (rad) del sensor de nest (receptor de radio r_i). 360°: la
señal del nest se percibe sin que el robot esté mirando hacia ella.
"""

FARMING_FOV_OBJECT_SENSOR: float = math.radians(120)
"""
Campo de visión (rad) del sensor de objetos del robot: cono frontal
centrado en su rumbo, como una cámara gran angular. Un objeto fuera del
cono no se detecta aunque esté dentro del radio.
"""

FARMING_EDGE_FOLLOW_WEIGHT: float = 0.4
"""
Peso (0-1) de la componente tangencial que se mezcla con la repulsión de
una parcela: 0 = solo alejarse, 1 = solo recorrer el borde. Favorece
bordear la parcela (obstáculo alargado) en vez de rebotar contra ella.
"""

EXPLORE_FREE_ITERS: int = 0
"""
Iteraciones sin ninguna zona activa durante las que el robot conserva su
último rumbo antes de empezar la exploración tipo vuelo de Lévy. Con 0 la
exploración empieza de inmediato: al arrancar, cada robot sale en una
dirección aleatoria.
"""

DIREXP_RESET_NOISE: float = 0.1
"""
Perturbación gaussiana (rad) que se suma al rumbo de exploración cuando el
robot pasa de exploración libre a tener alguna zona activa.
"""

FARMING_LEVY_MIN_RUN: int = 20
"""Duración mínima (iteraciones) de una racha recta del vuelo de Lévy."""

FARMING_LEVY_MAX_RUN: int = 300
"""Duración máxima (iteraciones) de una racha recta del vuelo de Lévy."""

FARMING_LEVY_EXPONENT: float = 1.5
"""
Exponente de la distribución de Pareto de la duración de cada racha:
mezcla rachas cortas frecuentes con rachas largas ocasionales (cola
pesada), típico del forrajeo animal (valores entre 1 y 3).
"""

FARMING_LEVY_TURN_NOISE_STD: float = 0.15 * 0.2
"""
Desviación estándar (rad, ~0.03) del ruido de rumbo en cada iteración de
una racha recta del vuelo de Lévy, para que no sea una línea perfecta.
"""

# ══════════════════════════════════════════════════════════════════════════════
# FARMING — ejecución del giro
# ══════════════════════════════════════════════════════════════════════════════
#
# El rumbo de cada robot se ajusta de forma cinemática en farming.run():
# normalmente se promedia 50/50 con la dirección RAOI de la iteración. Las
# constantes siguientes acotan ese ajuste cuando la dirección deseada es
# inestable; ninguna cambia la dirección que decide el modelo RAOI.

FARMING_TURN_JUMP_THRESHOLD: float = math.radians(70)
"""
Cambio de la dirección deseada entre dos iteraciones (rad) a partir del
cual se considera un salto inestable (objetivo puntual muy cerca, rebote,
parpadeo de zona entre vecinos) y el giro se limita a FARMING_TURN_CAP.
"""

FARMING_TURN_CAP: float = math.radians(20)
"""Giro máximo por iteración (rad) mientras la dirección deseada salta."""

FARMING_TURN_STREAK_LIMIT: int = 6
"""
Iteraciones seguidas con salto inestable tras las cuales se activa la
válvula de escape (FARMING_TURN_COOLDOWN). Rompe la persecución circular
que produce el tope de giro cuando el objetivo sigue adelantando al robot
en el mismo sentido.
"""

FARMING_TURN_COOLDOWN: int = 10
"""
Duración (iteraciones) de la válvula de escape: el robot sigue derecho con
su rumbo actual, sin girar, para cambiar la geometría robot-objetivo.
"""

FARMING_SPIN_WINDOW: int = 10
"""
Ventana (iteraciones) de la segunda condición de la válvula de escape:
giro sostenido sin avance. Se acumula el giro real del cuerpo y se mide la
dispersión de las posiciones dentro de la ventana. Detecta el giro sobre
el propio eje que producen dos vecinos cuya zona RAOI parpadea, que no
genera saltos consecutivos. Es corta para que el tramo recorrido antes de
un rebote salga pronto de la ventana.
"""

FARMING_SPIN_ROTATION_THRESHOLD: float = math.radians(150)
"""
Giro acumulado (rad) en FARMING_SPIN_WINDOW que, junto con
FARMING_SPIN_MAX_SPREAD, activa la válvula de escape. Los giros legítimos
(darse vuelta tras recoger un objeto) no superan este valor con
desplazamiento bajo.
"""

FARMING_SPIN_MAX_SPREAD: float = 0.6
"""
Dispersión máxima (m) de las posiciones en FARMING_SPIN_WINDOW —distancia
máxima de cualquiera de ellas a su centroide— para considerar que el robot
no avanza. No depende de la fase de una órbita, a diferencia del
desplazamiento neto entre los extremos de la ventana.
"""

FARMING_FINAL_APPROACH_RADIUS: float = 1.5
"""
Radio (m) de aproximación final a un objetivo puntual (objeto o nest).

Con el estímulo del objetivo a menos de esta distancia y la
influencia activa, el rumbo se fija directamente en la dirección RAOI —el
robot gira en su lugar— en vez de promediarse 50/50 con el rumbo actual.
Cerca de un punto, el promedio 50/50 es una persecución con ganancia 0.5
que tiene una órbita estable de ~0.5 m de radio, justo fuera del radio de
recolección. Debe ser mayor que el paso máximo por iteración más el radio
de recolección (~1.0 + 0.4 m).
"""

# ══════════════════════════════════════════════════════════════════════════════
# FARMING — ruido de sensores y actuadores
# ══════════════════════════════════════════════════════════════════════════════

FARMING_SENSOR_ANGLE_NOISE_STD: float = 0.03
"""Desviación estándar (rad, ~1.7°) del ruido en el ángulo percibido hacia un objeto o el nest."""

FARMING_SENSOR_DISTANCE_NOISE_STD: float = 0.03
"""
Desviación estándar (m) del ruido en la distancia percibida hacia un
objeto o el nest; hace difuso el borde de detección.
"""

FARMING_ACTUATOR_NOISE_STD: float = 0.03
"""
Desviación estándar (V) del ruido sumado al voltaje de las ruedas en cada
iteración (voltajes de operación ~0.8–2.0 V).
"""

# ══════════════════════════════════════════════════════════════════════════════
# FARMING — escenario
# ══════════════════════════════════════════════════════════════════════════════

FARMING_AREA_LIMITS: float = 100.0
"""Lado del área cuadrada de simulación (m)."""

FARMING_SPAWN_LINE_X: float = 0.5
"""
Posición X mínima (m) de la columna vertical donde arrancan los robots, todos
mirando hacia +X (hacia las parcelas). Nunca queda dentro del margen de la
pared oeste.
"""

FARMING_SPAWN_SPACING_FACTOR: float = 3.0
"""
Espaciado ideal entre robots al arrancar, en múltiplos del radio de
repulsión. Si no caben todos, se reduce hasta el mínimo seguro
(2 × radio de repulsión).
"""

SPAWN_MIN_SEPARATION: float = 0.3
"""Separación mínima absoluta (m) entre robots al arrancar."""

FARMING_PLOT_LENGTH: float = 70.0
"""
Longitud (m) de cada parcela (segmento horizontal). Las parcelas se
recorren hacia el borde este: con área de 100 m y margen este de 5 m
ocupan x ∈ [25, 95], y dejan un pasillo de 25 m del lado oeste, donde
están el arranque y el nest.
"""

FARMING_PLOT_EAST_MARGIN: float = 5.0
"""Colchón (m) entre el extremo este de las parcelas y la pared este."""

FARMING_PLOT_SEPARATION: float = 4.0
"""
Distancia (m) entre los ejes de parcelas adyacentes. Con
FARMING_PLOT_REPULSION = 0.6 el pasillo libre entre dos filas mide 2.8 m,
~3 veces el diámetro del robot.
"""

FARMING_PLOT_MARGIN_Y: float = 3.0
"""
Margen (m) sin parcelas en los bordes norte y sur. Junto con la separación
fija el número máximo de parcelas, que farming calcula y usa como valor
por defecto de n_plots.
"""

FARMING_PLOT_REPULSION: float = 0.6
"""
Radio (m) de la zona de repulsión de una parcela, medido desde su eje.
Debe ser mayor que el radio del robot escalado (0.45 m) y bastante menor
que la mitad de FARMING_PLOT_SEPARATION.
"""

FARMING_PLOT_VISUAL_THICKNESS: float = 0.5
"""Semiespesor (m) con que se dibuja la cama de cultivo; no afecta la física."""

FARMING_STRIP_GAP: float = 0.1
"""
Holgura (m) entre el borde de la zona de repulsión de una parcela y el
inicio de su franja de objetos, para que ningún objeto quede dentro de la
zona de repulsión.
"""

FARMING_STRIP_WIDTH: float = 1.1
"""
Ancho (m) de la franja de objetos a cada lado de una parcela. Con
FARMING_PLOT_REPULSION + FARMING_STRIP_GAP = 0.7 m, la franja termina a
1.8 m del eje, antes de la mitad del pasillo (2.0 m).
"""

FARMING_PLOT_CONTACT_GAP: float = 0.01
"""
Distancia (m) al eje de la parcela a la que queda un robot cuyo paso
intenta atravesarla. El contacto es deslizante: se cancela solo el
movimiento normal a la parcela y se conservan el avance a lo largo de
ella y la velocidad.
"""

FARMING_PLOT_CONTACT_DEFLECTION: float = math.radians(20)
"""
Ángulo (rad) respecto a la parcela con que queda el rumbo tras un contacto:
a lo largo de ella, ligeramente hacia afuera. Un robot cargando la recorre
hacia el lado del nest; los demás, en el sentido en que ya avanzaban en X.
"""

FARMING_PLOT_CONTACT_HEADING_NOISE_STD: float = 0.15
"""Desviación estándar (rad) del ruido sumado al rumbo tras un contacto con parcela."""

# ══════════════════════════════════════════════════════════════════════════════
# FARMING — nest, brújula y objetos
# ══════════════════════════════════════════════════════════════════════════════

FARMING_NEST_POSITION: list = [5.0, FARMING_AREA_LIMITS / 2]
"""
Posición [x, y] (m) del nest: un punto en el pasillo oeste, a la altura
del centro del área.
"""

FARMING_NEST_RI: float = 8.0
"""Radio sensor (m) del robot hacia el nest (componente receptora r_i)."""

FARMING_NEST_RS: float = 4.0
"""
Radio de emisión (m) del nest (componente emisora r_s). El nest se
detecta dentro de un círculo de radio FARMING_NEST_RI + FARMING_NEST_RS,
con oclusión por parcelas.
"""

FARMING_NEST_ICON_RADIUS: float = 1.8
"""Separación (m) entre el punto del nest y su etiqueta en la animación."""

FARMING_COMPASS_HEADING: float = math.pi
"""
Rumbo (rad) que indica la brújula hacia el lado de la arena donde está el
nest (pi = oeste). La brújula no apunta al nest: no da información
norte-sur ni distancia. Inspirada en la brújula solar de las hormigas del
desierto (Cataglyphis), es el comportamiento de menor prioridad: solo la
usa un robot cargado sin ninguna zona RAOI activa ni estímulo del nest.
"""

FARMING_COMPASS_ERROR_STD: float = math.radians(12)
"""
Desviación estándar (rad) del sesgo de la brújula. Cada robot sortea un
sesgo fijo al iniciar la simulación y lo conserva toda la corrida, como
una brújula ligeramente descalibrada. El sesgo afecta a todos los rumbos
que da la brújula (oeste, norte y sur).
"""

FARMING_COMPASS_NOISE_STD: float = math.radians(4)
"""Desviación estándar (rad) del ruido de cada lectura de la brújula."""

FARMING_SEARCH_LEG_ITERS: int = 15
"""
Duración (iteraciones) del primer tramo de la búsqueda en vaivén. Un robot
cargado que toca la cerca oeste sin percibir el nest lo busca a lo largo de
ella: avanza hacia el norte o el sur (al azar) un tramo de L, regresa a la
cerca oeste, invierte el sentido y avanza 2L, luego 3L, y así hasta
percibir el estímulo del nest. Un tramo termina antes si toca la cerca
norte o sur.
"""

FARMING_OBJECT_SENSOR_RADIUS: float = 1.0
"""
Radio r_d (m) del sensor de objetos de cada robot, medido desde el borde de
su carrocería: un objeto se detecta a menos de r_d de ella (a
ROBOT_BODY_RADIUS·FARMING_ROBOT_SCALE + r_d del centro), dentro del cono
FARMING_FOV_OBJECT_SENSOR y con línea de vista libre (oclusión por
parcelas). Los objetos no emiten señal: es un sensor distinto del de nest.
"""

FARMING_PICK_RADIUS: float = 0.4
"""Radio (m) de recolección: el robot recoge un objeto al pasar a esta distancia."""

FARMING_DEPOSIT_RADIUS: float = 0.5
"""
Radio (m) de depósito alrededor del punto del nest. El objeto queda en el
punto del tramo recorrido más cercano al nest.
"""

FARMING_SWEPT_CONTACT: bool = True
"""
True: recoger y depositar se evalúan sobre el tramo recorrido en la
iteración (el robot pasó físicamente por encima). False: solo en la
posición final, con lo que un paso de 0.4–1.0 m puede cruzar el círculo
de recolección sin recoger el objeto.
"""

# ══════════════════════════════════════════════════════════════════════════════
# FARMING — métricas
# ══════════════════════════════════════════════════════════════════════════════

FARMING_ENERGY_COST_SEARCH: float = 1.0
"""Costo energético por iteración buscando, sin objeto (unidades arbitrarias)."""

FARMING_ENERGY_COST_CARRY: float = 1.5
"""Costo energético por iteración cargando un objeto (unidades arbitrarias)."""

# Costo de señalización (f4): f4 = n·c_i·(r_i/r_0)^p + c_s·(r_s/r_0)^q, con
# r_i el radio sensor de cada robot (receptor) y r_s el radio de emisión del
# nest (emisor). Un solo emisor y un receptor por robot; el costo crece con
# el cuadrado del radio (ley del inverso del cuadrado de la señal).

FARMING_SIGNAL_COST_SENSOR: float = 1.0
"""Factor de costo c_i del sensor (receptor) de cada robot."""

FARMING_SIGNAL_COST_EMITTER: float = 5.0
"""Factor de costo c_s del emisor del nest; mayor que c_i (más caro a igual radio)."""

FARMING_SIGNAL_SENSOR_EXPONENT: float = 2.0
"""Exponente p con que crece el costo del sensor con su radio."""

FARMING_SIGNAL_EMITTER_EXPONENT: float = 2.0
"""Exponente q con que crece el costo del emisor con su radio."""

FARMING_SIGNAL_REFERENCE_RADIUS: float = 1.0
"""Radio de referencia r_0 (m) que hace a f4 adimensional."""
