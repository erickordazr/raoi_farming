"""Pruebas de caracterización: la simulación reproduce la referencia guardada.

Comparan, con semilla fija, el reporte del enjambre, las posiciones de los
objetos y las funciones objetivo contra ``tests/data/<escenario>.npz``.
Cualquier cambio de código que altere la dinámica las hace fallar; si el
cambio es intencional, la referencia se regenera con
``tests/data/generate_reference.py``.

La tolerancia (1e-9) absorbe diferencias de último bit entre plataformas
en las funciones trigonométricas, no cambios de comportamiento.
"""

from pathlib import Path

import numpy as np
import pytest
from scenarios import SCENARIOS, run_scenario

DATA_DIR = Path(__file__).resolve().parent / "data"
ATOL: float = 1e-9


@pytest.fixture(scope="module", params=sorted(SCENARIOS))
def scenario(request: pytest.FixtureRequest) -> tuple[str, tuple, dict]:
    """Ejecuta un escenario una sola vez por módulo y carga su referencia."""
    name = request.param
    reference = np.load(DATA_DIR / f"{name}.npz")
    return name, run_scenario(name), {key: reference[key] for key in reference.files}


def test_report_matches_reference(scenario: tuple[str, tuple, dict]) -> None:
    """Posiciones, rumbos, velocidades y zonas de los robots coinciden."""
    _, (report, _, _), reference = scenario
    assert report.shape == reference["report"].shape
    np.testing.assert_allclose(report, reference["report"], rtol=0.0, atol=ATOL)


def test_objects_match_reference(scenario: tuple[str, tuple, dict]) -> None:
    """Las posiciones de los objetos coinciden en cada cuadro muestreado."""
    _, (_, objects_report, _), reference = scenario
    np.testing.assert_allclose(objects_report, reference["objects_report"], rtol=0.0, atol=ATOL)


def test_objectives_match_reference(scenario: tuple[str, tuple, dict]) -> None:
    """Las funciones objetivo f1–f4 coinciden."""
    _, (_, _, metrics), reference = scenario
    np.testing.assert_allclose(metrics, reference["metrics"], rtol=0.0, atol=ATOL)


def test_same_seed_is_reproducible() -> None:
    """Dos ejecuciones con la misma semilla dan exactamente el mismo resultado."""
    first = run_scenario("standard")
    second = run_scenario("standard")
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
