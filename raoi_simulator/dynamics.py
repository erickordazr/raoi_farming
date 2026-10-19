"""Modelo dinámico del robot diferencial y su integrador numérico.

Modelo de robot de tracción diferencial de Ordaz-Rivas et al. (2018):
masa, inercia, geometría de ruedas y actuadores DC.

Estado del robot: c = [x, y, z, theta, v, omega]
  x, y   — posición del centroide (m)
  z      — sin uso en 2D (siempre 0)
  theta  — orientación (rad)
  v      — velocidad lineal (m/s)
  omega  — velocidad angular (rad/s)

Entradas: u = [u_r, u_l] — voltajes de las ruedas derecha e izquierda (V).

Referencia:
  Ordaz-Rivas et al. (2018). Collective Tasks for a Flock of Robots
  Using Influence Factor. J. Intelligent & Robotic Systems.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import math

import numpy as np

from . import config


class DynamicsConstants:
    """Coeficientes del modelo dinámico, precalculados una vez por simulación.

    Atributos usados por _derivatives():
        d      : Distancia del centroide al eje de ruedas (m).
        r      : Radio de rueda (m).
        md     : m · d (término de Coriolis).
        BKAi** : Elementos de B · Kl⁻¹ · A⁻¹ (amortiguamiento por fuerza contraelectromotriz).
        Mei**  : Elementos de la inversa de la masa efectiva.
        BKKs** : Elementos de B · Kl⁻¹ · Ks (ganancia voltaje → aceleración).

    Con B la matriz rueda → (v, ω) inversa y A la matriz cinemática del
    robot diferencial.
    """

    def __init__(
        self,
        mass:      float | None = None,
        inertia:   float | None = None,
        d:         float | None = None,
        wheel_r:   float | None = None,
        wheel_sep: float | None = None,
        motor_ts:  float | None = None,
        motor_ks:  float | None = None,
        motor_kl:  float | None = None,
    ) -> None:
        """Precalcula los coeficientes a partir de los parámetros físicos.

        Los parámetros que se dejen en None toman el valor del robot base de
        config.py.

        Args:
            mass: Masa del robot (kg).
            inertia: Momento de inercia respecto al centroide (kg·m²).
            d: Distancia del centroide al eje de las ruedas (m).
            wheel_r: Radio de rueda (m).
            wheel_sep: Semiseparación entre ruedas (m).
            motor_ts: Constante de tiempo del motor (s).
            motor_ks: Ganancia de velocidad del motor (rad / s·N·m).
            motor_kl: Ganancia de corriente del motor (rad / s·V).
        """
        m   = mass      if mass      is not None else config.ROBOT_MASS
        Im  = inertia   if inertia   is not None else config.ROBOT_INERTIA
        d   = d         if d         is not None else config.ROBOT_D
        r   = wheel_r   if wheel_r   is not None else config.ROBOT_WHEEL_R
        R   = wheel_sep if wheel_sep is not None else config.ROBOT_WHEEL_SEP
        Ts  = motor_ts  if motor_ts  is not None else config.MOTOR_Ts
        Ks  = motor_ks  if motor_ks  is not None else config.MOTOR_Ks
        Kl  = motor_kl  if motor_kl  is not None else config.MOTOR_Kl

        self.d  = d
        self.r  = r
        self.md = m * d

        # Matrices B (velocidades de rueda) y A (cinemática del diferencial).
        B11 = 1/r;  B12 = 1/r;  B21 = R/r;  B22 = -R/r
        A11 = r/2;  A12 = r/2;  A21 = r/(2*R);  A22 = -r/(2*R)

        det_A = A11*A22 - A12*A21
        Ai11 =  A22/det_A;  Ai12 = -A12/det_A
        Ai21 = -A21/det_A;  Ai22 =  A11/det_A

        kl_inv = 1.0 / Kl
        BKi11 = B11*kl_inv;  BKi12 = B12*kl_inv
        BKi21 = B21*kl_inv;  BKi22 = B22*kl_inv

        BKTs11 = BKi11*Ts;  BKTs12 = BKi12*Ts
        BKTs21 = BKi21*Ts;  BKTs22 = BKi22*Ts

        BKTsAi11 = BKTs11*Ai11 + BKTs12*Ai21
        BKTsAi12 = BKTs11*Ai12 + BKTs12*Ai22
        BKTsAi21 = BKTs21*Ai11 + BKTs22*Ai21
        BKTsAi22 = BKTs21*Ai12 + BKTs22*Ai22

        # Masa efectiva (cuerpo + actuadores) y su inversa.
        Me11 = m + BKTsAi11;  Me12 = BKTsAi12
        Me21 = BKTsAi21;      Me22 = (Im + m*d**2) + BKTsAi22

        det_Me     = Me11*Me22 - Me12*Me21
        self.Mei11 =  Me22/det_Me;  self.Mei12 = -Me12/det_Me
        self.Mei21 = -Me21/det_Me;  self.Mei22 =  Me11/det_Me

        self.BKAi11 = BKi11*Ai11 + BKi12*Ai21
        self.BKAi12 = BKi11*Ai12 + BKi12*Ai22
        self.BKAi21 = BKi21*Ai11 + BKi22*Ai21
        self.BKAi22 = BKi21*Ai12 + BKi22*Ai22

        ks_kl = Ks * kl_inv
        self.BKKs11 = B11 * ks_kl
        self.BKKs12 = B12 * ks_kl
        self.BKKs21 = B21 * ks_kl
        self.BKKs22 = B22 * ks_kl


def _derivatives(C: np.ndarray, U: np.ndarray, dyn: DynamicsConstants) -> np.ndarray:
    """dC/dt para todos los robots a la vez (vectorizado).

    Args:
        C:   Estado de los robots, shape (N, 6): [x, y, z, theta, v, omega].
        U:   Voltajes [u_r, u_l], shape (N, 2).
        dyn: Coeficientes del modelo.

    Returns:
        dC/dt, shape (N, 6).
    """
    theta = C[:, 3]
    v     = C[:, 4]
    omega = C[:, 5]

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    dx = cos_t * v  - dyn.d * sin_t * omega
    dy = sin_t * v  + dyn.d * cos_t * omega

    H1 = -dyn.md * omega**2
    H2 =  dyn.md * v * omega

    BKKs_u1 = dyn.BKKs11 * U[:, 0] + dyn.BKKs12 * U[:, 1]
    BKKs_u2 = dyn.BKKs21 * U[:, 0] + dyn.BKKs22 * U[:, 1]

    BKAi_v1 = dyn.BKAi11 * v + dyn.BKAi12 * omega
    BKAi_v2 = dyn.BKAi21 * v + dyn.BKAi22 * omega

    rhs1 = BKKs_u1 - (H1 + BKAi_v1)
    rhs2 = BKKs_u2 - (H2 + BKAi_v2)

    dv     = dyn.Mei11 * rhs1 + dyn.Mei12 * rhs2
    domega = dyn.Mei21 * rhs1 + dyn.Mei22 * rhs2

    dC = np.zeros_like(C)
    dC[:, 0] = dx
    dC[:, 1] = dy
    dC[:, 3] = omega
    dC[:, 4] = dv
    dC[:, 5] = domega
    return dC


def step(
    C: np.ndarray,
    U: np.ndarray,
    dyn: DynamicsConstants,
    dt: float = config.DT,
) -> np.ndarray:
    """Integra el modelo dinámico de todos los robots un paso dt con RK4.

    El paso se divide en config.RK4_SUBSTEPS subpasos.

    Args:
        C:   Estado de los robots, shape (N, 6).
        U:   Voltajes [u_r, u_l], shape (N, 2).
        dyn: Coeficientes del modelo.
        dt:  Duración del paso (s).

    Returns:
        Estado en t + dt, shape (N, 6).
    """
    h  = dt / config.RK4_SUBSTEPS
    Cn = C.copy()
    for _ in range(config.RK4_SUBSTEPS):
        k1 = _derivatives(Cn,            U, dyn)
        k2 = _derivatives(Cn + h/2 * k1, U, dyn)
        k3 = _derivatives(Cn + h/2 * k2, U, dyn)
        k4 = _derivatives(Cn + h   * k3, U, dyn)
        Cn = Cn + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return Cn


def resolve_walls(
    c_old: np.ndarray,
    c_new: np.ndarray,
    margin: float,
    area_limits: float,
    v_max: float,
) -> tuple[np.ndarray, bool]:
    """Aplica los límites de velocidad y el contacto con las paredes del área.

    Contacto: si el paso deja al robot a menos de `margin` de una pared,
      - rumbo: si el robot apuntaba hacia el interior se refleja la
        componente que cruzó; si apuntaba hacia la pared, se redirige al
        centro del área con ruido (config.WALL_BOUNCE_HEADING_NOISE_STD);
      - posición: deslizamiento. Se conserva el desplazamiento del paso y
        solo la coordenada que cruzó vuelve al margen
        (más config.WALL_CONTACT_GAP).

    Args:
        c_old:       Estado del robot antes del paso, shape (6,).
        c_new:       Estado integrado por step(), shape (6,); se modifica
                     en sitio.
        margin:      Distancia mínima a las paredes (m): el radio de
                     repulsión del robot.
        area_limits: Lado del área cuadrada (m).
        v_max:       Velocidad lineal máxima (m/s).

    Returns:
        c_new  : Estado final, shape (6,).
        bounced: True si hubo contacto con una pared.
    """
    c_new[4] = float(np.clip(c_new[4],  0.0,              v_max))
    c_new[5] = float(np.clip(c_new[5], -config.OMEGA_MAX, config.OMEGA_MAX))

    hit_x = (c_new[0] < margin or c_new[0] > area_limits - margin)
    hit_y = (c_new[1] < margin or c_new[1] > area_limits - margin)

    if hit_x or hit_y:
        vx_dir = math.cos(c_old[3])
        vy_dir = math.sin(c_old[3])
        dx_in  = area_limits / 2.0 - c_old[0]
        dy_in  = area_limits / 2.0 - c_old[1]

        if vx_dir * dx_in + vy_dir * dy_in <= 0:
            angle_to_center = math.atan2(dy_in, dx_in)
            c_new[3] = (
                angle_to_center + np.random.normal(0.0, config.WALL_BOUNCE_HEADING_NOISE_STD)
            ) % (2 * math.pi)
        else:
            if hit_x:
                vx_dir = -vx_dir
            if hit_y:
                vy_dir = -vy_dir
            c_new[3] = math.atan2(vy_dir, vx_dir) % (2 * math.pi)

        inner = margin + config.WALL_CONTACT_GAP
        if c_new[0] < margin:
            c_new[0] = inner
        if c_new[0] > area_limits - margin:
            c_new[0] = area_limits - inner
        if c_new[1] < margin:
            c_new[1] = inner
        if c_new[1] > area_limits - margin:
            c_new[1] = area_limits - inner

    c_new[3] = c_new[3] % (2 * math.pi)
    return c_new, bool(hit_x or hit_y)
