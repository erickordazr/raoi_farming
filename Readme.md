# RAOI Farming Simulator

Simulador de enjambre de robots con el modelo **RAOI** (Repulsión, Atracción,
Orientación, Influencia) aplicado a la tarea de **farming**: un enjambre de
robots debe recolectar objetos distribuidos en franjas a los costados de
parcelas agrícolas y depositarlos en el nido.

Basado en:

> Ordaz-Rivas et al. (2018). *Collective Tasks for a Flock of Robots Using
> Influence Factor*. J. Intelligent & Robotic Systems.
>
> Ordaz-Rivas et al. (2021). *Autonomous foraging with a pack of robots based
> on repulsion, attraction and influence*. Autonomous Robots.

## Características

- Modelo dinámico de robot diferencial (masa, inercia, motor DC) integrado con RK4.
- Comportamiento RAOI puro: repulsión, orientación, atracción e influencia
  compuestas por prioridad, sin waypoints ni lógica de navegación explícita.
- Robot escalado físicamente (dimensiones, masa, inercia y velocidad
  máxima) para aproximarse a un robot agrícola real, sin afectar los
  parámetros base del modelo.
- Sensado realista: radios de detección cortos (tipo cámara/beacon) y
  oclusión de visión — un robot no detecta un objeto ni el nido si una
  parcela se interpone físicamente en la línea de percepción.
- Colisión física dura contra las parcelas (no solo repulsión blanda).
- Exploración libre tipo vuelo de Lévy (rachas rectas de duración variable
  + giros grandes ocasionales) cuando ningún estímulo está activo.
- Métricas de desempeño **f1**, **f2** y **f3** (balance de carga, gasto
  energético del enjambre y fracción de objetos entregados).
- Animación con Pygame (+ grabación de video opcional con OpenCV).

## Instalación

```bash
pip install -r Requirements.txt
```

Requiere Python 3.10+ (usa sintaxis de type hints con `|` y `tuple[...]`).

## Uso

### Modo interactivo

```bash
python main.py
```

Muestra un menú con dos opciones: correr una simulación individual o una
tanda de réplicas con reporte estadístico. Los parámetros se piden por
consola; dejar en blanco los avanzados (radios de detección, ventana de
tiempo fija) usa los valores por defecto de `config.py`.

### Desde código

```python
from raoi_simulator.farming import single_run, statistical_run

# Una simulación con animación
report, objects_report, metrics = single_run(
    n_objects=10, n_plots=3, individuals=15,
    r_r=0.1, o_r=0.5, a_r=1.5,
    animation=True,
)

# Múltiples réplicas con tabla estadística
metrics_report, running_mean, final_mean = statistical_run(
    replicas=10,
    n_objects=10, n_plots=3, individuals=15,
    r_r=0.1, o_r=0.5, a_r=1.5,
)
```

O directamente con `run()` para control total (incluye radios de detección
y ventana de tiempo fija):

```python
from raoi_simulator.farming import run

report, objects_report, metrics = run(
    n_objects=10, n_plots=3, individuals=15,
    r_r=0.1, o_r=0.5, a_r=1.5,
    animation=False,
    seed=42,
    nest_ri=10.0, nest_rs=2.0,   # radio de detección del nido: ri + rs
    obj_r=1.0,                   # radio de detección de un objeto (cámara)
    time_limit=7200,             # ventana de tiempo fija (None = usa el default)
)
```

## Métricas

| Métrica | Descripción |
|---|---|
| **f1** | Balance de carga: objetos recolectados por cada robot (`objects_per_robot`) y su desviación estándar (`load_balance_std`). 0 = trabajo perfectamente balanceado. |
| **f2** | Gasto energético del enjambre: +1 por iteración en búsqueda, +1.5 por iteración transportando un objeto. Se reporta por robot (`energy_per_robot`), total (`total_energy`) y promedio (`mean_energy`). |
| **f3** | Fracción de objetos entregados sobre el total (`success_fraction`) dentro de la ventana de iteraciones disponible. |

## Parámetros principales

Todos viven en `raoi_simulator/config.py`, agrupados y documentados por sección.

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `FARMING_AREA_LIMITS` | 100 m | Lado del área cuadrada de simulación. |
| `FARMING_ROBOT_SCALE` | 6.0 | Factor de escala física del robot (dimensiones, masa, inercia, velocidad). |
| `FARMING_PLOT_LENGTH` | 75 m | Longitud de cada parcela. |
| `FARMING_PLOT_SEPARATION` | 4 m | Distancia centro a centro entre parcelas adyacentes. |
| `FARMING_NEST_RI` / `FARMING_NEST_RS` | 10 / 2 m | Radio sensor del robot / radio de emisión del nido (se suman). |
| `FARMING_OBJECT_DETECTION_RADIUS` | 1 m | Radio de detección de un objeto (sensor tipo cámara, sin emisión). |
| `FARMING_DEFAULT_MAX_ITER` | 7200 | Iteraciones por defecto de una simulación (1 iteración = 1 s simulado). |

Con `FARMING_AREA_LIMITS=100`, el número máximo de parcelas que caben en el
área es 17 (calculado por `_compute_max_plots()`); un valor moderado como
`n_plots=3` es un buen punto de partida para pruebas.

## Estructura del proyecto

```
raoi_simulator/
├── config.py         # Todos los parámetros del modelo, robot, escenario y visualización
├── behavior.py        # Reglas RAOI: repulsión, orientación, atracción, composición, voltaje
├── dynamics.py         # Modelo dinámico del robot diferencial + integrador RK4
├── environment.py      # Detección de paredes y parcelas como fuentes de repulsión
├── farming.py           # Lógica de la tarea: run(), single_run(), statistical_run()
├── metrics.py            # Cálculo de f1, f2, f3
└── visualization.py       # Animación Pygame + grabación de video
main.py                     # Punto de entrada interactivo
Requirements.txt
```

## Autor

Erick Ordaz-Rivas <erick.ordazrv@uanl.edu.mx>
FIME — Universidad Autónoma de Nuevo León