"""Genera los archivos de referencia de las pruebas de caracterización.

Solo debe ejecutarse cuando un cambio del modelo altera los resultados a
propósito; en cualquier otro caso las pruebas deben pasar con la referencia
existente.

Uso, desde la raíz del repositorio:

    python tests/data/generate_reference.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from scenarios import SCENARIOS, run_scenario  # noqa: E402


def main() -> None:
    """Ejecuta cada escenario y guarda su referencia en ``tests/data/<nombre>.npz``."""
    out_dir = Path(__file__).resolve().parent
    for name in SCENARIOS:
        report, objects_report, metrics = run_scenario(name)
        np.savez_compressed(
            out_dir / f"{name}.npz", report=report, objects_report=objects_report, metrics=metrics
        )
        print(f"{name}: report {report.shape}, objetos {objects_report.shape}, f1–f4 {metrics}")


if __name__ == "__main__":
    main()
