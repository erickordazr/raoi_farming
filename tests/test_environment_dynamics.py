"""Pruebas unitarias del entorno (``environment``) y de la dinámica (``dynamics``)."""

import math

import numpy as np
import pytest

from raoi_simulator import config, dynamics, environment

PLOT = {"x0": 0.0, "x1": 10.0, "y": 5.0}


def test_detect_walls_corner_returns_two_points() -> None:
    """En una esquina se detectan dos paredes."""
    points = environment.detect_walls(np.array([0.1, 0.1]), 0.5, 100.0)
    assert sorted(map(tuple, points)) == [(0.0, 0.1), (0.1, 0.0)]


def test_detect_walls_center_returns_nothing() -> None:
    """Lejos de las paredes no se detecta ninguna."""
    assert environment.detect_walls(np.array([50.0, 50.0]), 0.5, 100.0) == []


def test_detect_plots_closest_point() -> None:
    """El punto de la parcela es la proyección del robot sobre el segmento."""
    assert environment.detect_plots(np.array([3.0, 5.4]), [PLOT], 0.5) == [[3.0, 5.0]]
    assert environment.detect_plots(np.array([3.0, 6.0]), [PLOT], 0.5) == []


def test_segment_blocked_is_symmetric() -> None:
    """La oclusión no depende del orden de los extremos."""
    a, b = np.array([5.0, 4.0]), np.array([6.0, 6.0])
    assert environment.segment_blocked_by_plots(a, b, [PLOT])
    assert environment.segment_blocked_by_plots(b, a, [PLOT])


def test_segment_same_y_or_beside_plot_is_not_blocked() -> None:
    """Una línea horizontal o que pasa fuera del segmento no está bloqueada."""
    blocked = environment.segment_blocked_by_plots
    assert not blocked(np.array([0, 4.0]), np.array([9, 4.0]), [PLOT])
    assert not blocked(np.array([12, 4.0]), np.array([12, 6.0]), [PLOT])


def test_equal_voltages_reach_terminal_speed_without_rotation() -> None:
    """Con voltajes iguales el robot avanza recto y alcanza v = Ks · r · u."""
    dyn = dynamics.DynamicsConstants()
    C = np.zeros((1, 6))
    U = np.array([[1.5, 1.5]])
    for _ in range(30):
        C = dynamics.step(C, U, dyn)
    assert C[0, 5] == pytest.approx(0.0, abs=1e-12)
    assert C[0, 4] == pytest.approx(config.MOTOR_Ks * config.ROBOT_WHEEL_R * 1.5, rel=1e-3)
    assert C[0, 1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(("x", "y", "theta"), [(0.05, 50.0, math.pi), (99.95, 50.0, 0.0)])
def test_resolve_walls_keeps_robot_inside(x: float, y: float, theta: float) -> None:
    """Un paso que cruzaría una pared deja al robot dentro del área."""
    margin, area = 0.3, 100.0
    c_old = np.array([50.0 + (x - 50.0) * 0.99, y, 0.0, theta, 1.0, 0.0])
    c_new = np.array([x, y, 0.0, theta, 1.0, 0.0])
    fixed, bounced = dynamics.resolve_walls(c_old, c_new.copy(), margin, area, 1.0)
    assert bounced
    assert margin <= fixed[0] <= area - margin
