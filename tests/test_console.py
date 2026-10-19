"""Pruebas del menú por consola y de los resultados que guardan las corridas."""

import builtins
from pathlib import Path

import numpy as np
import pytest

from raoi_simulator import config, console, farming

EMPTY_PARAMS = dict.fromkeys(
    ("n_objects", "n_plots", "individuals", "r_r", "o_r", "a_r", "r_r_deliver",
     "o_r_deliver", "a_r_deliver", "nest_ri", "nest_rs", "obj_r", "time_limit")
)


def test_prompt_order_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con respuestas vacías se usan los valores por defecto, en el orden acordado."""
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    monkeypatch.setattr(builtins, "input", fake_input)
    p = console.prompt_parameters(dict(EMPTY_PARAMS), max_plots=23)

    expected_start = [
        "Number of objects", "Number of individuals", "Number of farm rows",
        "Repulsion radius, search phase", "Orientation radius, search phase",
        "Attraction radius, search phase", "Repulsion radius, delivery phase",
        "Orientation radius, delivery phase", "Attraction radius, delivery phase",
        "Nest sensor radius", "Nest emission radius", "Object sensor radius",
        "Simulation time",
    ]
    assert all(q.startswith(e) for q, e in zip(prompts, expected_start, strict=True))
    assert p["n_objects"] == config.FARMING_DEFAULT_N_OBJECTS
    assert p["individuals"] == config.FARMING_DEFAULT_INDIVIDUALS
    assert p["n_plots"] == 23
    assert (p["r_r"], p["o_r"], p["a_r"]) == (
        config.FARMING_DEFAULT_R_REPULSION,
        config.FARMING_DEFAULT_R_ORIENTATION,
        config.FARMING_DEFAULT_R_ATTRACTION,
    )
    assert (p["r_r_deliver"], p["o_r_deliver"], p["a_r_deliver"]) == (p["r_r"], p["o_r"], p["a_r"])
    assert p["time_limit"] == config.FARMING_DEFAULT_MAX_ITER


def test_single_run_saves_results_folder(tmp_path: Path) -> None:
    """single_run guarda los .npy y el reporte de texto en una carpeta propia."""
    farming.single_run(
        n_objects=12, n_plots=2, individuals=4, r_r=0.3, o_r=1.0, a_r=2.0, animation=False,
        r_r_deliver=0.3, o_r_deliver=1.0, a_r_deliver=2.0, nest_ri=8.0, nest_rs=4.0,
        obj_r=1.0, time_limit=40, save_dir=tmp_path,
    )
    (run_dir,) = list(tmp_path.iterdir())
    assert run_dir.name.endswith("_single")
    names = {f.name for f in run_dir.iterdir()}
    assert names == {"farming_report.npy", "farming_objects_report.npy", config.REPORT_FILENAME}
    text = (run_dir / config.REPORT_FILENAME).read_text(encoding="utf-8")
    for label in ("Number of objects", "Repulsion radius, search phase",
                  "Nest emission radius, r_s", "f1 Load balance", "f4 Signaling cost"):
        assert label in text
    assert np.load(run_dir / "farming_report.npy").shape == (40, 4, 8)


def test_statistical_run_saves_results_folder(tmp_path: Path) -> None:
    """statistical_run guarda las funciones objetivo por réplica y su reporte de texto."""
    metrics_report, _, _ = farming.statistical_run(
        2, n_objects=12, n_plots=2, individuals=4, r_r=0.3, o_r=1.0, a_r=2.0,
        r_r_deliver=0.3, o_r_deliver=1.0, a_r_deliver=2.0, nest_ri=8.0, nest_rs=4.0,
        obj_r=1.0, time_limit=30, save_dir=tmp_path,
    )
    (run_dir,) = list(tmp_path.iterdir())
    assert run_dir.name.endswith("_stats")
    np.testing.assert_array_equal(np.load(run_dir / "farming_metrics_report.npy"), metrics_report)
    text = (run_dir / config.REPORT_FILENAME).read_text(encoding="utf-8")
    assert "Objective functions per replica" in text
    assert f"Seeds: {config.SEED} to {config.SEED + 1}" in text


def test_no_files_without_save_dir(tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Con save_dir=None no se escribe ningún archivo."""
    monkeypatch.chdir(tmp_path)
    farming.single_run(
        n_objects=12, n_plots=2, individuals=4, r_r=0.3, o_r=1.0, a_r=2.0, animation=False,
        r_r_deliver=0.3, o_r_deliver=1.0, a_r_deliver=2.0, nest_ri=8.0, nest_rs=4.0,
        obj_r=1.0, time_limit=20, save_dir=None,
    )
    assert list(tmp_path.iterdir()) == []


def test_animation_defaults_to_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dejar en blanco la pregunta de animación responde YES; NO la desactiva."""
    answers = iter(["", "no"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(answers))
    assert console.ask_yes_no("Animation? ", None, default=True) is True
    assert console.ask_yes_no("Animation? ", None, default=True) is False
