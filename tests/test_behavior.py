"""Pruebas unitarias de las reglas RAOI (``raoi_simulator.behavior``)."""

import math

import numpy as np
import pytest

from raoi_simulator import behavior, config

FOV_ALL = {
    "fov_repulsion": 2 * math.pi,
    "fov_orientation": 2 * math.pi,
    "fov_attraction": 2 * math.pi,
}
WEIGHTS = {"w_r": 0.8, "w_o": 0.2, "w_a": 0.15, "w_I": 0.4}


def _state(*robots: tuple[float, float, float]) -> np.ndarray:
    """Construye un estado (N, 6) a partir de tuplas (x, y, theta)."""
    C = np.zeros((len(robots), 6))
    for k, (x, y, theta) in enumerate(robots):
        C[k, 0], C[k, 1], C[k, 3] = x, y, theta
    return C


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.0), (2 * math.pi, 0.0), (-math.pi / 2, 1.5 * math.pi), (5 * math.pi, math.pi)],
)
def test_wrap_angle_range(angle: float, expected: float) -> None:
    """El ángulo normalizado queda en [0, 2π)."""
    assert behavior.wrap_angle(angle) == pytest.approx(expected)


def test_angle_diff_sign_and_range() -> None:
    """La diferencia es la mínima con signo, en [-π, π)."""
    assert behavior.angle_diff(0.1, 2 * math.pi - 0.1) == pytest.approx(0.2)
    assert behavior.angle_diff(2 * math.pi - 0.1, 0.1) == pytest.approx(-0.2)
    diffs = behavior.angle_diff(np.linspace(-10, 10, 101), 0.0)
    assert np.all(diffs >= -math.pi) and np.all(diffs < math.pi)


def test_repulsion_vector_points_away() -> None:
    """Con una sola fuente, el vector es unitario y se aleja de ella."""
    vx, vy = behavior.repulsion_vector(np.array([1.0, 0.0]), [[0.0, 0.0]])
    assert (vx, vy) == pytest.approx((1.0, 0.0))


def test_repulsion_vector_overlapping_source_is_finite() -> None:
    """Una fuente en la misma posición no produce división entre cero."""
    vx, vy = behavior.repulsion_vector(np.array([1.0, 1.0]), [[1.0, 1.0]])
    assert math.isfinite(vx) and math.isfinite(vy)


def test_combined_direction_without_zones_keeps_heading() -> None:
    """Sin zonas activas se conserva el rumbo actual."""
    assert behavior.combined_direction(1.2, {}, WEIGHTS) == 1.2


def test_combined_direction_single_zone_has_no_inertia() -> None:
    """Con una sola zona activa la dirección es exactamente la de esa zona."""
    theta = behavior.combined_direction(0.0, {"R": (0.0, 1.0)}, WEIGHTS)
    assert theta == pytest.approx(math.pi / 2)


def test_detect_neighbors_zones_by_distance() -> None:
    """Cada vecino cae en la zona que le corresponde por distancia."""
    C = _state((0, 0, 0), (1.5, 0, 0), (2.5, 0, 0))
    out = behavior.detect_neighbors(0, C, 1.0, 2.0, 3.0, FOV_ALL, 0, [])
    assert (out["n_rep"], out["n_ori"], out["n_att"]) == (0, 1, 1)


def test_detect_neighbors_repulsion_overrides_other_zones() -> None:
    """Con un vecino en repulsión no se evalúan orientación ni atracción."""
    C = _state((0, 0, 0), (0.5, 0, 0), (1.5, 0, 0), (2.5, 0, 0))
    out = behavior.detect_neighbors(0, C, 1.0, 2.0, 3.0, FOV_ALL, 0, [])
    assert (out["n_rep"], out["n_ori"], out["n_att"]) == (1, 0, 0)


def test_detect_neighbors_wall_contact_disables_orientation_and_attraction() -> None:
    """Una pared o parcela en contacto anula orientación y atracción."""
    C = _state((0, 0, 0), (1.5, 0, 0), (2.5, 0, 0))
    out = behavior.detect_neighbors(0, C, 1.0, 2.0, 3.0, FOV_ALL, 1, [])
    assert (out["n_ori"], out["n_att"]) == (0, 0)


def test_detect_neighbors_respects_field_of_view() -> None:
    """Un vecino detrás del robot no cuenta en una zona con campo de visión frontal."""
    C = _state((0, 0, 0), (-0.5, 0, 0))
    fov = dict(FOV_ALL, fov_repulsion=math.pi)
    out = behavior.detect_neighbors(0, C, 1.0, 2.0, 3.0, fov, 0, [])
    assert out["n_rep"] == 0


def test_detect_neighbors_occlusion_by_plot() -> None:
    """Una parcela entre dos robots bloquea la percepción."""
    C = _state((0, 0, 0), (0, 1.5, 0))
    plots = [{"x0": -5.0, "x1": 5.0, "y": 0.75}]
    out = behavior.detect_neighbors(0, C, 1.0, 2.0, 3.0, FOV_ALL, 0, plots)
    assert out["n_ori"] == 0


def test_select_voltage_by_zone() -> None:
    """Cada zona usa su voltaje de referencia, sin ruido."""
    v = {"repulsion": 1.0, "orientation": 2.0, "attraction": 3.0}
    assert behavior.select_voltage({"R": (1, 0)}, 0.0, 1.0, 0.0, v).tolist() == [1.0, 1.0]
    assert behavior.select_voltage({"A": (1, 0)}, 0.0, 1.0, 0.0, v).tolist() == [3.0, 3.0]
    assert behavior.select_voltage({}, 0.0, 1.0, 0.0, v).tolist() == [2.0, 2.0]


def test_select_voltage_influence_slows_down_near_target() -> None:
    """Con influencia, el voltaje crece con la distancia al estímulo."""
    v = {"repulsion": 1.0, "orientation": 2.0, "attraction": 3.0}
    near = behavior.select_voltage({"I": (1, 0)}, 0.1, 1.0, 0.0, v)[0]
    far = behavior.select_voltage({"I": (1, 0)}, 0.9, 1.0, 0.0, v)[0]
    assert v["repulsion"] <= near < far <= v["attraction"]


def test_weights_in_config_are_complete() -> None:
    """La configuración define los cuatro pesos que usa combined_direction."""
    assert set(config.FARMING_RAOI_WEIGHTS) == set(WEIGHTS)
