"""Configuración común de pytest.

Agrega la raíz del repositorio y la carpeta ``tests`` a ``sys.path`` para
que las pruebas importen ``raoi_simulator`` y ``scenarios`` sin instalar el
paquete, y fuerza el modo sin pantalla de SDL por si alguna prueba carga
la visualización.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
