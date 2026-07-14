"""
Modelo dinámico del robot diferencial y su integrador numérico.

Implementa el modelo de robot de tracción diferencial descrito en
Ordaz-Rivas et al. (2018), incluyendo masa, inercia, geometría de ruedas
y características del actuador DC.

Estado del robot: c = [x, y, z, theta, v, omega]
  x, y    — posición del centroide (m)
  z       — reservado (siempre 0 en simulación 2D)
  theta   — orientación (rad)
  v       — velocidad lineal (m/s)
  omega   — velocidad angular (rad/s)

Entradas: u = [u_r, u_l] — voltajes en rueda derecha e izquierda (V)

Referencia:
  Ordaz-Rivas et al. (2018). Collective Tasks for a Flock of Robots
  Using Influence Factor. J. Intelligent & Robotic Systems.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

import numpy as np

from . import config


class DynamicsConstants:
    """
    Constantes escalares del modelo dinámico precalculadas.

    Todas las matrices del modelo dependen exclusivamente de parámetros
    físicos fijos. Instanciar una sola vez por simulación y pasar la
    instancia a step() evita recalcular ~40 operaciones por robot por
    iteración durante la integración RK4.

    Atributos:
        d, r, R     : Parámetros geométricos del robot.
        md          : m * d (producto usado en Coriolis).
        Ai**        : Elementos de la inversa de la matriz cinemática A.
        BKi**       : Elementos de B @ Kl_inv.
        BKAi_**     : Elementos de B @ Kl_inv @ A_inv.
        Mei**       : Elementos de la inversa de la masa efectiva M_eff.
        ks_kl       : Ks / Kl (escalar).
        BKKs**      : Elementos de B @ Kl_inv @ Ks.
    """

    def __init__(
        self,
        mass:      float = None,
        inertia:   float = None,
        d:         float = None,
        wheel_r:   float = None,
        wheel_sep: float = None,
        motor_ts:  float = None,
        motor_ks:  float = None,
        motor_kl:  float = None,
    ) -> None:
        """
        Args:
            mass, inertia, d, wheel_r, wheel_sep, motor_ts, motor_ks, motor_kl:
                Overrides opcionales de los parametros fisicos del robot.
                Cualquiera que se deje en None toma el valor global de
                config.py. Permite instanciar un robot fisicamente distinto
                (por ejemplo, un robot escalado en tamaño y masa para una
                tarea especifica) sin afectar a otras tareas que instancian
                DynamicsConstants() sin argumentos.
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
        self.R  = R
        self.md = m * d

        # Matrices de la cinemática (B, A) y sus productos
        B11 = 1/r;  B12 = 1/r;  B21 = R/r;  B22 = -R/r
        A11 = r/2;  A12 = r/2;  A21 = r/(2*R);  A22 = -r/(2*R)

        det_A     = A11*A22 - A12*A21
        self.Ai11 =  A22/det_A;  self.Ai12 = -A12/det_A
        self.Ai21 = -A21/det_A;  self.Ai22 =  A11/det_A

        kl_inv = 1.0 / Kl
        BKi11 = B11*kl_inv;  BKi12 = B12*kl_inv
        BKi21 = B21*kl_inv;  BKi22 = B22*kl_inv

        BKTs11 = BKi11*Ts;  BKTs12 = BKi12*Ts
        BKTs21 = BKi21*Ts;  BKTs22 = BKi22*Ts

        BKTsAi11 = BKTs11*self.Ai11 + BKTs12*self.Ai21
        BKTsAi12 = BKTs11*self.Ai12 + BKTs12*self.Ai22
        BKTsAi21 = BKTs21*self.Ai11 + BKTs22*self.Ai21
        BKTsAi22 = BKTs21*self.Ai12 + BKTs22*self.Ai22

        Me11 = m + BKTsAi11;  Me12 = BKTsAi12
        Me21 = BKTsAi21;       Me22 = (Im + m*d**2) + BKTsAi22

        det_Me      = Me11*Me22 - Me12*Me21
        self.Mei11  =  Me22/det_Me;  self.Mei12 = -Me12/det_Me
        self.Mei21  = -Me21/det_Me;  self.Mei22 =  Me11/det_Me

        self.BKAi11 = BKi11*self.Ai11 + BKi12*self.Ai21
        self.BKAi12 = BKi11*self.Ai12 + BKi12*self.Ai22
        self.BKAi21 = BKi21*self.Ai11 + BKi22*self.Ai21
        self.BKAi22 = BKi21*self.Ai12 + BKi22*self.Ai22

        self.ks_kl  = Ks * kl_inv
        self.BKKs11 = B11 * self.ks_kl
        self.BKKs12 = B12 * self.ks_kl
        self.BKKs21 = B21 * self.ks_kl
        self.BKKs22 = B22 * self.ks_kl


def _derivatives(C: np.ndarray, U: np.ndarray, dyn: DynamicsConstants) -> np.ndarray:
    """
    Calcula dC/dt para todos los robots simultáneamente (vectorizado).

    Implementa las ecuaciones de movimiento del robot diferencial incluyendo
    dinámica de actuadores DC (modelo de armadura de inductancia pequeña).

    Args:
        C:   Estado de todos los robots, shape (N, 6).
             Columnas: [x, y, z, theta, v, omega].
        U:   Voltajes de entrada [u_r, u_l], shape (N, 2).
        dyn: Constantes del modelo precalculadas.

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
    """
    Integra el modelo dinámico un paso de tiempo dt con RK4 vectorizado.

    Opera sobre todos los robots en paralelo usando NumPy. El paso dt
    se subdivide en config.RK4_SUBSTEPS pasos internos para mayor
    precisión numérica.

    Args:
        C:   Estado actual de todos los robots, shape (N, 6).
        U:   Voltajes de entrada [u_r, u_l], shape (N, 2).
        dyn: Constantes del modelo precalculadas (ver DynamicsConstants).
        dt:  Paso de tiempo total en segundos (default config.DT).

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


def integrate_robot(
    c_i: np.ndarray,
    voltage: np.ndarray,
    dyn: DynamicsConstants,
    repulsion_radius: float,
    area_limits: float,
    v_max_linear: float = None,
    omega_max: float = None,
) -> tuple[np.ndarray, bool]:
    """
    Integra la dinámica de un robot individual y gestiona colisiones con paredes.

    Aplica la integración RK4, satura las velocidades a los límites físicos
    del Khepera III, y aplica un rebote en paredes con corrección de orientación.

    Estrategia de rebote:
      - Si el robot ya apunta hacia el interior (dot > 0): reflexión especular.
      - Si apunta hacia la pared (dot ≤ 0): redirigir al centro del área
        con una pequeña perturbación aleatoria para romper simetrías.
      - En ambos casos, empujar la posición al margen para evitar que
        el siguiente paso vuelva a cruzar el límite.

    Args:
        c_i:              Estado actual del robot, shape (6,).
        voltage:          Voltajes de entrada [u_r, u_l], shape (2,).
        dyn:              Constantes del modelo precalculadas.
        repulsion_radius: Radio de repulsión efectivo (margen de pared, m).
        area_limits:      Lado del área cuadrada (m).
        v_max_linear:     Override opcional de velocidad lineal maxima (m/s).
                          None -> usa config.V_MAX_LINEAR.
        omega_max:        Override opcional de velocidad angular maxima (rad/s).
                          None -> usa config.OMEGA_MAX.

    Returns:
        c_new  : Estado actualizado, shape (6,).
        bounced: True si hubo colisión con una pared.
    """
    import math

    v_max = v_max_linear if v_max_linear is not None else config.V_MAX_LINEAR
    om_max = omega_max   if omega_max    is not None else config.OMEGA_MAX

    C1    = c_i.reshape(1, 6)
    U1    = voltage.reshape(1, 2)
    c_new = step(C1, U1, dyn)[0]

    c_new[4] = float(np.clip(c_new[4],  0.0,     v_max))
    c_new[5] = float(np.clip(c_new[5], -om_max,  om_max))

    hit_x = (c_new[0] < repulsion_radius or c_new[0] > area_limits - repulsion_radius)
    hit_y = (c_new[1] < repulsion_radius or c_new[1] > area_limits - repulsion_radius)

    if hit_x or hit_y:
        c_new = c_i.copy()

        vx_dir = math.cos(c_new[3])
        vy_dir = math.sin(c_new[3])
        dx_in  = area_limits / 2.0 - c_new[0]
        dy_in  = area_limits / 2.0 - c_new[1]
        dot    = vx_dir * dx_in + vy_dir * dy_in

        if dot <= 0:
            angle_to_center = math.atan2(dy_in, dx_in)
            c_new[3] = (angle_to_center + np.random.normal(0.0, 0.2)) % (2 * math.pi)
        else:
            if hit_x:
                vx_dir = -vx_dir
            if hit_y:
                vy_dir = -vy_dir
            c_new[3] = math.atan2(vy_dir, vx_dir) % (2 * math.pi)

        escape = repulsion_radius + 0.01
        if c_new[0] < repulsion_radius:               c_new[0] = escape
        if c_new[0] > area_limits - repulsion_radius: c_new[0] = area_limits - escape
        if c_new[1] < repulsion_radius:               c_new[1] = escape
        if c_new[1] > area_limits - repulsion_radius: c_new[1] = area_limits - escape

    c_new[3] = c_new[3] % (2 * math.pi)
    return c_new, bool(hit_x or hit_y)