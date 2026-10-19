# RAOI Farming Simulator

Simulador de enjambre de robots con el modelo **RAOI** (Repulsión, Atracción,
Orientación, Influencia) aplicado a la tarea de **farming**: un enjambre de
robots recolecta objetos distribuidos en franjas a los costados de parcelas
agrícolas y los deposita en el nido.

Basado en:

> Ordaz-Rivas et al. (2018). *Collective Tasks for a Flock of Robots Using
> Influence Factor*. J. Intelligent & Robotic Systems.
>
> Ordaz-Rivas et al. (2021). *Autonomous foraging with a pack of robots based
> on repulsion, attraction and influence*. Autonomous Robots.

## Características

- Modelo dinámico de robot diferencial (masa, inercia, motor DC) integrado con RK4.
- Comportamiento RAOI: repulsión, orientación, atracción e influencia
  compuestas por prioridad, sin waypoints ni navegación explícita.
- Robot escalado como un vehículo agrícola (~0.9 m de diámetro, ~82 kg), con
  el motor escalado por similitud dinámica: responde tan rápido como el robot
  de laboratorio.
- Velocidad de avance por zona RAOI: repulsión 0.4 m/s, orientación y
  exploración 0.7 m/s, atracción y tránsito 1.0 m/s; con influencia, el robot
  desacelera al acercarse al objetivo.
- Dos sensores por robot, con oclusión (un robot no percibe un objeto, el
  nido ni a otro robot si una parcela se interpone en la línea de vista):
  sensor de nido de 360° con radio r_i (el nido emite con radio r_s y se
  percibe dentro de r_i + r_s) y sensor de objetos tipo cámara, cono frontal
  de 120° con radio r_d medido desde el borde del robot (los objetos no
  emiten señal).
- Regreso con brújula bioinspirado (hormigas del desierto): un robot cargado
  que no percibe nada solo sabe que el nido está hacia el oeste (con sesgo fijo
  y ruido por lectura). Si toca la cerca oeste sin percibir el nido, lo busca
  en vaivén creciente a lo largo de ella. La brújula tiene menor prioridad que
  cualquier zona RAOI.
- Contacto deslizante con paredes y parcelas; recolección y depósito
  evaluados sobre el tramo recorrido en cada iteración.
- Exploración libre tipo vuelo de Lévy cuando ninguna zona está activa.
- Funciones objetivo f1–f4: balance de carga, energía promedio, tasa de
  éxito y costo de señalización (sensores r_i de los robots y emisor r_s del
  nido).
- Animación con Pygame o grabación de video con OpenCV; cada robot dibuja su
  sensor activo (`SHOW_SENSOR_RADIUS`): el cono de objetos r_d si va sin carga
  o el círculo de nido r_i si va cargado.

## Instalación

Requiere Python 3.10 o posterior.

```bash
pip install -r requirements.txt
```

Paquetes que se instalan:

| Paquete | Versión mínima | Para qué se usa |
|---|---|---|
| `numpy` | 1.24 | Cálculo numérico, estado del enjambre y generadores aleatorios |
| `pygame` | 2.1.3 | Animación (ventana o cuadros del video) |
| `opencv-python` | 4.7 | Escritura del video mp4 |
| `tqdm` | 4.65 | Barras de progreso |

Para desarrollo (pruebas y análisis estático) se instala además
`requirements-dev.txt`, que agrega `pytest`, `ruff`, `mypy` y `pre-commit`:

```bash
pip install -r requirements-dev.txt
```

## Pruebas

```bash
pytest
```

Las pruebas de caracterización comparan una simulación con semilla fija
contra la referencia guardada en `tests/data/`. Si un cambio del modelo
altera los resultados a propósito, la referencia se regenera con
`python tests/data/generate_reference.py`.

## Uso

### Modo interactivo

```bash
python main.py
```

Primero se elige la opción del menú (simulación individual o tanda de
réplicas; para las réplicas se pide además cuántas). Después se piden los
parámetros en este orden; dejar la respuesta en blanco usa el valor por
defecto:

| # | Pregunta en consola | Argumento | Por defecto |
|---|---|---|---|
| 1 | Number of objects | `n_objects` | 250 |
| 2 | Number of individuals (robots) | `individuals` | 25 |
| 3 | Number of farm rows (plots) | `n_plots` | máximo que cabe (23) |
| 4 | Repulsion radius, search phase (m) | `r_r` | 0.25 |
| 5 | Orientation radius, search phase (m) | `o_r` | 1.0 |
| 6 | Attraction radius, search phase (m) | `a_r` | 2.0 |
| 7 | Repulsion radius, delivery phase (m) | `r_r_deliver` | el de búsqueda |
| 8 | Orientation radius, delivery phase (m) | `o_r_deliver` | el de búsqueda |
| 9 | Attraction radius, delivery phase (m) | `a_r_deliver` | el de búsqueda |
| 10 | Nest sensor radius of each robot, r_i (m) | `nest_ri` | 8.0 |
| 11 | Nest emission radius, r_s (m) | `nest_rs` | 4.0 |
| 12 | Object sensor radius of each robot, from its edge, r_d (m) | `obj_r` | 1.0 |
| 13 | Simulation time (iterations, 1 iteration = 1 s) | `time_limit` | 7200 |
| 14 | Animation? (YES/NO), solo en la simulación individual | `animation` | YES |

La fase de búsqueda es cuando el robot no lleva objeto; la de entrega,
cuando lo lleva al nido. Los valores por defecto están en `config.py`
(`FARMING_DEFAULT_*`, `FARMING_NEST_RI`, `FARMING_NEST_RS`,
`FARMING_OBJECT_SENSOR_RADIUS`). Todos los radios están en metros. Los radios
RAOI y el sensor de objetos se miden desde el borde del robot; el sensor de
nido y el emisor, desde su centro. Los valores se validan antes de simular:
conteos de al menos 1, radios no negativos y `r_r <= o_r <= a_r` en cada fase.

### Resultados

Cada corrida crea su propia carpeta dentro de `results/`, con fecha y hora:

```
results/
├── 2026-10-19_14-30-05_single/
│   ├── report.txt                  # parámetros y funciones objetivo f1–f4
│   ├── farming_report.npy          # estado del enjambre por iteración (T, N, 8)
│   ├── farming_objects_report.npy  # posiciones de los objetos por iteración (T, O, 2)
│   └── simulation.mp4              # video de la animación (si se pidió)
└── 2026-10-19_15-02-41_stats/
    ├── report.txt                  # parámetros, f1–f4 por réplica, media y desviación
    ├── farming_metrics_report.npy  # [f1, f2, f3, f4] por réplica (R, 4)
    ├── farming_running_mean.npy    # media acumulada tras cada réplica (R, 4)
    └── farming_final_mean.npy      # media de todas las réplicas (4,)
```

La carpeta `results/` no se sube a GitHub (`.gitignore`). Desde código,
`save_dir` cambia la carpeta raíz de resultados y `save_dir=None` no guarda
archivos. Con `SAVE_VIDEO = False` en `config.py` la animación se muestra en
una ventana en lugar de grabarse.

### Desde código

```python
from raoi_simulator.farming import single_run, statistical_run

# Una simulación con animación
report, objects_report, metrics = single_run(
    n_objects=60, n_plots=6, individuals=16,
    r_r=0.3, o_r=1.0, a_r=2.0,
    animation=True,
)

# Varias réplicas con tabla estadística
metrics_report, running_mean, final_mean = statistical_run(
    replicas=10,
    n_objects=60, n_plots=6, individuals=16,
    r_r=0.3, o_r=1.0, a_r=2.0,
)
```

O directamente con `run()`, sin preguntas por consola:

```python
from raoi_simulator.farming import run

report, objects_report, metrics = run(
    n_objects=60, n_plots=6, individuals=16,
    r_r=0.3, o_r=1.0, a_r=2.0,
    animation=False,
    seed=42,
    nest_ri=8.0, nest_rs=4.0,   # el nido se detecta dentro de ri + rs
    obj_r=1.0,                  # sensor de objetos, desde el borde del robot
    time_limit=7200,            # iteraciones máximas (None = valor por defecto)
)
```

`report` tiene forma (T, N, 8): `[x, y, z, theta (rad), theta (grados), v,
omega, zona activa]`, con zona 0 = exploración libre, 1 = repulsión,
2 = atracción, 3 = orientación, 4 = influencia.

## Funciones objetivo

| Clave | Función | Sentido | Descripción |
|---|---|---|---|
| `load_balance` | f1 | minimizar | Desviación estándar de los objetos recogidos por cada robot (0 = trabajo perfectamente repartido). |
| `mean_energy` | f2 | minimizar | Energía total entre el número de robots: 1 por iteración buscando, 1.5 por iteración cargando un objeto. |
| `success_ratio` | f3 | maximizar | Objetos entregados entre objetos disponibles dentro de las iteraciones disponibles. |
| `signaling_cost` | f4 | minimizar | Costo de señalización: n·c_i·(r_i/r_0)² + c_s·(r_s/r_0)², sensores de los robots (receptores, radio r_i) más el emisor del nido (radio r_s); c_i = 1, c_s = 5, r_0 = 1 m (`FARMING_SIGNAL_*` en `config.py`). |

`single_run()` reporta f1–f4 de una simulación; `statistical_run()` reporta
su media y desviación estándar sobre las réplicas y guarda
`farming_metrics_report.npy` con forma (R, 4) = [f1, f2, f3, f4].

## Parámetros principales

Todos están en `raoi_simulator/config.py`, agrupados y documentados por sección.

| Parámetro | Valor | Descripción |
|---|---|---|
| `FARMING_AREA_LIMITS` | 100 m | Lado del área cuadrada. |
| `FARMING_ROBOT_SCALE` | 6.0 | Escala del robot respecto al de laboratorio. |
| `FARMING_ZONE_SPEEDS` | 0.4 / 0.7 / 1.0 m/s | Velocidad de avance en repulsión / orientación / atracción. |
| `FARMING_PLOT_LENGTH` | 70 m | Longitud de cada parcela. |
| `FARMING_PLOT_SEPARATION` | 4 m | Distancia entre ejes de parcelas adyacentes. |
| `FARMING_NEST_RI` / `FARMING_NEST_RS` | 8 / 4 m | Radio sensor del robot / radio de emisión del nido (se suman). |
| `FARMING_OBJECT_SENSOR_RADIUS` | 1 m | Radio r_d del sensor de objetos del robot, desde el borde de su carrocería. |
| `FARMING_FOV_OBJECT_SENSOR` / `FARMING_FOV_NEST_SENSOR` | 120° / 360° | Campo de visión del sensor de objetos / de nido. |
| `FARMING_DEFAULT_MAX_ITER` | 7200 | Iteraciones máximas por defecto (1 iteración = 1 s). |

Con los valores por defecto caben hasta 23 parcelas; ese es también el valor
por defecto de `n_plots` en el modo interactivo.

## Estructura del proyecto

```
raoi_simulator/
├── config.py          # Parámetros del modelo, robot, escenario y visualización
├── behavior.py        # Reglas RAOI: zonas de vecinos, composición de vectores, voltaje
├── dynamics.py        # Modelo dinámico del robot diferencial, RK4 y contacto con paredes
├── environment.py     # Paredes, parcelas y oclusión de la línea de vista
├── farming.py         # Tarea de farming: run(), single_run(), statistical_run()
├── console.py         # Preguntas por consola, tablas de resultados y barras de progreso
├── metrics.py         # Funciones objetivo f1–f4
└── visualization.py   # Animación Pygame y grabación de video
tests/                 # Pruebas unitarias y de caracterización (pytest)
├── data/              # Referencias con semilla fija de las pruebas de caracterización
results/               # Resultados de cada corrida (se crea al ejecutar; no se versiona)
_local/                # Notas y documentos personales (no se versiona)
main.py                # Menú interactivo
pyproject.toml         # Metadatos y configuración de ruff, mypy y pytest
requirements.txt       # Dependencias de ejecución
requirements-dev.txt   # Dependencias de desarrollo
```

## Autor

Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
FIME — Universidad Autónoma de Nuevo León
