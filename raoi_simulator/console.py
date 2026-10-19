"""Entrada y salida por consola del simulador.

Preguntas de parámetros, tablas de resultados y adaptación de las barras de
progreso de tqdm. El modelo (farming.run) no lee ni escribe en la consola;
solo single_run() y statistical_run() usan este módulo.

Autores: Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
         FIME — Universidad Autónoma de Nuevo León
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import config

_TABLE_WIDTH = 54   # ancho de las tablas de resultados (caracteres)


def ask(prompt: str, value: Any, cast: type, error: str, optional: bool = False) -> Any:
    """Devuelve value si no es None; si no, lo pide por consola hasta que sea válido.

    Args:
        prompt:   Texto a mostrar.
        value:    Valor ya dado (se devuelve tal cual si no es None).
        cast:     Tipo al que convertir la respuesta (int o float).
        error:    Mensaje si la respuesta no es válida.
        optional: Si True, una respuesta vacía devuelve None.

    Returns:
        El valor dado, el leído, o None (respuesta vacía con optional=True).
    """
    if value is not None:
        return value
    while True:
        raw = input(prompt)
        if optional and raw.strip() == "":
            return None
        try:
            return cast(raw.strip() if optional else raw)
        except ValueError:
            print(error)


def ask_yes_no(prompt: str, value: bool | None, default: bool | None = None) -> bool:
    """Devuelve value si no es None; si no, pide YES o NO por consola.

    Args:
        prompt:  Texto a mostrar.
        value:   Valor ya dado.
        default: Respuesta si se deja en blanco; None = la respuesta es obligatoria.

    Returns:
        True para YES, False para NO.
    """
    if value is not None:
        return value
    while True:
        ans = input(prompt).strip().upper()
        if ans == "" and default is not None:
            return default
        if ans in ("YES", "NO"):
            return ans == "YES"
        print("  Enter YES or NO.")


def prompt_parameters(p: dict, max_plots: int) -> dict:
    """Completa por consola los parámetros de simulación que vengan en None.

    Orden de las preguntas: objetos, robots, parcelas, radios RAOI de la fase
    de búsqueda, radios RAOI de la fase de entrega, sensor de nest r_i,
    emisor del nest r_s, sensor de objetos r_d y tiempo de simulación. Una
    respuesta vacía usa el valor por defecto que se muestra entre corchetes
    (config.FARMING_DEFAULT_*, los radios de búsqueda para los de entrega y
    el máximo de parcelas que caben).

    Args:
        p:         Parámetros de run() (sin animation ni seed); se modifica en sitio.
        max_plots: Número máximo de parcelas que caben en el área.

    Returns:
        El mismo dict p, completo.
    """
    def ask_value(key: str, label: str, default: float, cast: type) -> None:
        error = (f"  Invalid input. Enter {'an integer' if cast is int else 'a number'} "
                 f"or leave blank.")
        value  = ask(f"{label} [Enter = {default}]: ", p[key], cast, error, optional=True)
        p[key] = default if value is None else value

    ask_value("n_objects", "Number of objects", config.FARMING_DEFAULT_N_OBJECTS, int)
    ask_value("individuals", "Number of individuals (robots)",
              config.FARMING_DEFAULT_INDIVIDUALS, int)
    ask_value("n_plots", f"Number of farm rows (plots, max {max_plots})", max_plots, int)

    ask_value("r_r", "Repulsion radius, search phase (m)",
              config.FARMING_DEFAULT_R_REPULSION, float)
    ask_value("o_r", "Orientation radius, search phase (m)",
              config.FARMING_DEFAULT_R_ORIENTATION, float)
    ask_value("a_r", "Attraction radius, search phase (m)",
              config.FARMING_DEFAULT_R_ATTRACTION, float)

    ask_value("r_r_deliver", "Repulsion radius, delivery phase (m)",   p["r_r"], float)
    ask_value("o_r_deliver", "Orientation radius, delivery phase (m)", p["o_r"], float)
    ask_value("a_r_deliver", "Attraction radius, delivery phase (m)",  p["a_r"], float)

    ask_value("nest_ri", "Nest sensor radius of each robot, r_i (m)",
              config.FARMING_NEST_RI, float)
    ask_value("nest_rs", "Nest emission radius, r_s (m)", config.FARMING_NEST_RS, float)
    ask_value("obj_r", "Object sensor radius of each robot, from its edge, r_d (m)",
              config.FARMING_OBJECT_SENSOR_RADIUS, float)
    ask_value("time_limit", "Simulation time (iterations, 1 iteration = 1 s)",
              config.FARMING_DEFAULT_MAX_ITER, int)
    return p


def print_results(
    metrics:    dict,
    elapsed:    float,
    iterations: int,
    max_iter:   int,
    p:          dict,
) -> None:
    """Imprime la tabla de resultados de una simulación.

    Args:
        metrics:    Funciones objetivo devueltas por run().
        elapsed:    Tiempo de ejecución (s).
        iterations: Iteraciones ejecutadas (puede ser < max_iter si se
                    entregaron todos los objetos).
        max_iter:   Iteraciones máximas.
        p:          Parámetros de la simulación (ver prompt_parameters).
    """
    sep     = "=" * _TABLE_WIDTH
    ms_iter = elapsed / max(iterations, 1) * 1000
    search  = (f"r_repulsion={p['r_r']:.3f}  r_orientation={p['o_r']:.3f}  "
               f"r_attraction={p['a_r']:.3f}")
    deliver = (f"r_repulsion={p['r_r_deliver']:.3f}  r_orientation={p['o_r_deliver']:.3f}  "
               f"r_attraction={p['a_r_deliver']:.3f}")

    print(f"\n{sep}")
    print("  RAOI -- Farming Task - Results")
    print(sep)
    print(f"  N robots      : {p['individuals']:>4}      Farm rows    : {p['n_plots']:>6}")
    print(f"  Objects       : {p['n_objects']:>4}      Iterations   : {iterations} / {max_iter}")
    print(f"  [busqueda] {search}")
    print(f"  [entrega]  {deliver}")
    print(sep)
    print(f"  f1 Load balance  (min) : {metrics['load_balance']:>10.3f}")
    print(f"  f2 Mean energy   (min) : {metrics['mean_energy']:>10.3f}")
    print(f"  f3 Success ratio (max) : {metrics['success_ratio']:>10.3f}")
    print(f"  f4 Signal cost   (min) : {metrics['signaling_cost']:>10.3f}")
    print(sep)
    print(f"  Runtime : {elapsed:.2f} s   ({ms_iter:.1f} ms/iter)")
    print(f"{sep}\n")


def print_stats(metrics_report: np.ndarray, elapsed: float, replicas: int) -> None:
    """Imprime la tabla estadística de varias réplicas.

    Args:
        metrics_report: Shape (R, 4): [f1, f2, f3, f4] por réplica.
        elapsed:        Tiempo total de ejecución (s).
        replicas:       Número de réplicas (al menos 1).
    """
    sep  = "=" * _TABLE_WIDTH
    mean = np.mean(metrics_report, axis=0)
    std  = np.std(metrics_report,  axis=0)

    print(f"\n{sep}")
    print(f"  RAOI -- Farming - Statistical Results ({replicas} replicas)")
    print(sep)
    print(f"  {'Objective':<24} {'Mean':>10}  {'Std':>10}")
    print(f"  {'-' * 24} {'-' * 10}  {'-' * 10}")
    print(f"  {'f1 Load balance  (min)':<24} {mean[0]:>10.3f}  {std[0]:>10.3f}")
    print(f"  {'f2 Mean energy   (min)':<24} {mean[1]:>10.3f}  {std[1]:>10.3f}")
    print(f"  {'f3 Success ratio (max)':<24} {mean[2]:>10.3f}  {std[2]:>10.3f}")
    print(f"  {'f4 Signal cost   (min)':<24} {mean[3]:>10.3f}  {std[3]:>10.3f}")
    print(sep)
    print(f"  Avg runtime / replica   : {elapsed / max(replicas, 1):.2f} s")
    print(f"  Total runtime           : {elapsed:.2f} s")
    print(f"{sep}\n")


# ── Reportes de texto ─────────────────────────────────────────────────────────

_PARAMETER_LABELS: tuple[tuple[str, str], ...] = (
    ("n_objects",   "Number of objects"),
    ("individuals", "Number of individuals (robots)"),
    ("n_plots",     "Number of farm rows (plots)"),
    ("r_r",         "Repulsion radius, search phase (m)"),
    ("o_r",         "Orientation radius, search phase (m)"),
    ("a_r",         "Attraction radius, search phase (m)"),
    ("r_r_deliver", "Repulsion radius, delivery phase (m)"),
    ("o_r_deliver", "Orientation radius, delivery phase (m)"),
    ("a_r_deliver", "Attraction radius, delivery phase (m)"),
    ("nest_ri",     "Nest sensor radius of each robot, r_i (m)"),
    ("nest_rs",     "Nest emission radius, r_s (m)"),
    ("obj_r",       "Object sensor radius of each robot, from its edge, r_d (m)"),
    ("time_limit",  "Simulation time (iterations, 1 iteration = 1 s)"),
)
"""Parámetros de la corrida, en el orden en que se piden, con su nombre completo."""

_OBJECTIVE_LABELS: tuple[tuple[str, str], ...] = (
    ("load_balance",   "f1 Load balance (min)"),
    ("mean_energy",    "f2 Mean energy (min)"),
    ("success_ratio",  "f3 Success ratio (max)"),
    ("signaling_cost", "f4 Signaling cost (min)"),
)
"""Funciones objetivo f1–f4: clave en el dict de métricas y nombre completo."""


def _parameter_lines(p: dict) -> list[str]:
    """Líneas de texto con los parámetros de la corrida.

    Args:
        p: Parámetros de la simulación (ver prompt_parameters).

    Returns:
        Una línea por parámetro, con nombre completo y valor.
    """
    lines = ["Parameters"]
    for key, label in _PARAMETER_LABELS:
        lines.append(f"  {label:<60} {p[key]}")
    return lines


def _write_lines(path: Path, lines: list[str]) -> None:
    """Escribe líneas de texto en UTF-8.

    Args:
        path:  Archivo de salida.
        lines: Líneas sin salto final.
    """
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_report(
    path:       Path,
    p:          dict,
    metrics:    dict,
    iterations: int,
    elapsed:    float,
    seed:       int,
    files:      list[str],
) -> None:
    """Escribe el reporte de texto de una simulación: parámetros y funciones objetivo.

    Args:
        path:       Archivo de salida.
        p:          Parámetros de la simulación (ver prompt_parameters).
        metrics:    Funciones objetivo devueltas por run().
        iterations: Iteraciones ejecutadas.
        elapsed:    Tiempo de ejecución (s).
        seed:       Semilla aleatoria de la corrida.
        files:      Descripción de los archivos guardados junto al reporte.
    """
    delivered = round(metrics["success_ratio"] * p["n_objects"])
    lines = [
        "RAOI Swarm Simulator - Farming Task",
        "Single simulation report",
        f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Seed: {seed}",
        "",
        *_parameter_lines(p),
        "",
        "Objective functions",
        *(f"  {label:<60} {metrics[key]:.6f}" for key, label in _OBJECTIVE_LABELS),
        "",
        "Run",
        f"  {'Iterations executed':<60} {iterations} / {p['time_limit']}",
        f"  {'Objects delivered':<60} {delivered} / {p['n_objects']}",
        f"  {'Runtime (s)':<60} {elapsed:.2f}",
        "",
        "Files",
        *(f"  {item}" for item in files),
    ]
    _write_lines(path, lines)


def write_stats_report(
    path:           Path,
    p:              dict,
    metrics_report: np.ndarray,
    elapsed:        float,
    seeds:          list[int],
    files:          list[str],
) -> None:
    """Escribe el reporte de texto de varias réplicas: parámetros, f1–f4 por réplica y estadística.

    Args:
        path:           Archivo de salida.
        p:              Parámetros de la simulación (ver prompt_parameters).
        metrics_report: Shape (R, 4): [f1, f2, f3, f4] por réplica.
        elapsed:        Tiempo total de ejecución (s).
        seeds:          Semilla de cada réplica.
        files:          Descripción de los archivos guardados junto al reporte.
    """
    header = "  " + f"{'Replica':>7}  {'Seed':>6}  " + "  ".join(
        f"{label.split()[0]:>14}" for _, label in _OBJECTIVE_LABELS
    )
    rows = [
        "  " + f"{r + 1:>7}  {seed:>6}  " + "  ".join(f"{v:>14.6f}" for v in metrics_report[r])
        for r, seed in enumerate(seeds)
    ]
    mean = np.mean(metrics_report, axis=0)
    std  = np.std(metrics_report, axis=0)
    lines = [
        "RAOI Swarm Simulator - Farming Task",
        f"Statistical report ({len(seeds)} replicas)",
        f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Seeds: {seeds[0]} to {seeds[-1]}",
        "",
        *_parameter_lines(p),
        "",
        "Objective functions (mean and standard deviation over replicas)",
        *(f"  {label:<40} mean {mean[k]:>14.6f}   std {std[k]:>14.6f}"
          for k, (_, label) in enumerate(_OBJECTIVE_LABELS)),
        "",
        "Objective functions per replica",
        header,
        *rows,
        "",
        "Run",
        f"  {'Total runtime (s)':<60} {elapsed:.2f}",
        f"  {'Average runtime per replica (s)':<60} {elapsed / max(len(seeds), 1):.2f}",
        "",
        "Files",
        *(f"  {item}" for item in files),
    ]
    _write_lines(path, lines)


class TqdmProgress:
    """Adapta una barra de tqdm a la firma progress_callback(t, max_iter, delivered) de run().

    Attributes:
        bar:       Barra de tqdm que se actualiza.
        n_objects: Número total de objetos (para mostrar entregados / total).
        last_t:    Última iteración reportada.
    """

    def __init__(self, bar: Any, n_objects: int) -> None:
        """Inicializa el adaptador y muestra 0 objetos entregados.

        Args:
            bar:       Barra de tqdm.
            n_objects: Número total de objetos.
        """
        self.bar       = bar
        self.n_objects = n_objects
        self.last_t    = 0
        bar.set_postfix(obj=f"0/{n_objects}")

    def __call__(self, t: int, total: int, delivered: int) -> None:
        """Avanza la barra hasta la iteración t y muestra los objetos entregados.

        Args:
            t:         Iteración actual (desde 0).
            total:     Iteraciones máximas (no se usa: la barra ya conoce su total).
            delivered: Objetos entregados hasta ahora.
        """
        self.bar.update(t - self.last_t)
        self.bar.set_postfix(obj=f"{delivered}/{self.n_objects}")
        self.last_t = t

    def finish(self, iterations: int, delivered: int) -> None:
        """Completa la barra con las iteraciones ejecutadas y la cierra.

        Args:
            iterations: Iteraciones ejecutadas.
            delivered:  Objetos entregados al final.
        """
        self.bar.update(iterations - self.last_t)
        self.bar.set_postfix(obj=f"{delivered}/{self.n_objects}")
        self.bar.close()
