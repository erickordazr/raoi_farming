"""Pruebas de las funciones objetivo y de invariantes de la simulación."""

import io
from contextlib import redirect_stdout

import numpy as np
import pytest

from raoi_simulator import config, farming, metrics


def _counters(collected: list[int], carry: int = 0, search: int = 10) -> np.ndarray:
    """Contadores (N, 3) [iteraciones cargando, buscando, objetos recogidos]."""
    return np.array([[carry, search, c] for c in collected], dtype=float)


def test_load_balance_is_zero_for_uniform_work() -> None:
    """f1 = 0 cuando todos los robots recogen lo mismo."""
    out = metrics.farming_metrics(_counters([3, 3, 3]), 9, np.ones(9, bool), 8.0, 4.0)
    assert out["load_balance"] == 0.0


def test_success_ratio_and_energy() -> None:
    """f3 = entregados / total y f2 = energía media por robot."""
    delivered = np.array([True, True, False, False])
    out = metrics.farming_metrics(_counters([1, 1], carry=4, search=6), 4, delivered, 8.0, 4.0)
    assert out["success_ratio"] == 0.5
    expected_energy = 6 * config.FARMING_ENERGY_COST_SEARCH + 4 * config.FARMING_ENERGY_COST_CARRY
    assert out["mean_energy"] == pytest.approx(expected_energy)


def test_signaling_cost_formula() -> None:
    """f4 = n · r_i² + 5 · r_s² con los factores por defecto (16, 8, 4 → 1104)."""
    out = metrics.farming_metrics(_counters([0] * 16), 1, np.zeros(1, bool), 8.0, 4.0)
    assert out["signaling_cost"] == pytest.approx(1104.0)


def test_point_segment_distance() -> None:
    """Distancia de puntos a un segmento y punto más cercano."""
    dist, closest = farming._point_segment_distance(
        np.array([[0.5, 1.0], [2.0, 0.0]]), np.array([0.0, 0.0]), np.array([1.0, 0.0])
    )
    np.testing.assert_allclose(dist, [1.0, 1.0])
    np.testing.assert_allclose(closest, [[0.5, 0.0], [1.0, 0.0]])


def test_plots_are_symmetric_and_fit() -> None:
    """Las parcelas son simétricas respecto al centro y el máximo cabe en el área."""
    area = config.FARMING_AREA_LIMITS
    n_max = farming._compute_max_plots(
        area, config.FARMING_PLOT_SEPARATION, config.FARMING_STRIP_WIDTH,
        config.FARMING_PLOT_REPULSION, config.FARMING_PLOT_MARGIN_Y,
    )
    plots = farming._build_plots_symmetric(
        n_max, config.FARMING_PLOT_LENGTH, area, config.FARMING_PLOT_SEPARATION,
        config.FARMING_PLOT_EAST_MARGIN,
    )
    ys = np.array([p["y"] for p in plots])
    assert len(plots) == n_max
    np.testing.assert_allclose(np.sort(ys), np.sort(area - ys))
    assert ys.min() > 0 and ys.max() < area


@pytest.fixture(scope="module")
def short_run() -> tuple[np.ndarray, np.ndarray, dict]:
    """Simulación corta del escenario estándar."""
    with redirect_stdout(io.StringIO()):
        return farming.run(
            n_objects=60, n_plots=6, individuals=16, r_r=0.3, o_r=1.0, a_r=2.0,
            animation=False, seed=3, time_limit=300,
        )


def test_robots_stay_inside_area(short_run: tuple) -> None:
    """Ningún robot sale del área."""
    report = short_run[0]
    area = config.FARMING_AREA_LIMITS
    assert np.all((report[:, :, :2] >= 0.0) & (report[:, :, :2] <= area))


def test_number_of_objects_is_conserved(short_run: tuple) -> None:
    """El número de objetos no cambia y todos quedan dentro del área."""
    objects_report = short_run[1]
    assert objects_report.shape[1] == 60
    assert np.all(np.isfinite(objects_report))


def test_delivered_objects_are_at_the_nest(short_run: tuple) -> None:
    """Los objetos dentro del círculo de depósito al final son exactamente los entregados."""
    _, objects_report, out = short_run
    nest = np.array(config.FARMING_NEST_POSITION)
    at_nest = np.linalg.norm(objects_report[-1] - nest, axis=1) <= config.FARMING_DEPOSIT_RADIUS
    assert at_nest.sum() == round(out["success_ratio"] * 60)


def test_single_robot_runs() -> None:
    """La simulación funciona con un solo robot."""
    with redirect_stdout(io.StringIO()):
        report, objects_report, _ = farming.run(
            n_objects=4, n_plots=1, individuals=1, r_r=0.3, o_r=1.0, a_r=2.0,
            animation=False, seed=0, time_limit=50,
        )
    assert report.shape == (50, 1, 8)
    assert objects_report.shape == (50, 4, 2)
