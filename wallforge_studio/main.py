#!/usr/bin/env python3
"""
WallForge Studio — punto de entrada.

Uso:
    python3 main.py
    python3 main.py proyecto.wfp    # abre directamente un proyecto
"""
import sys
from pathlib import Path

# Garantizar que el paquete sea localizable independientemente del cwd
sys.path.insert(0, str(Path(__file__).parent.parent))

from wallforge_studio.ui.app import WallForgeApp
from wallforge_studio.model.project import Project


def main() -> None:
    app = WallForgeApp()

    # Si se pasa un archivo de proyecto como argumento, cargarlo al arrancar
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.exists():
            try:
                app.project = Project.load(path)
                app.canvas.project = app.project
                app.canvas.snap.grid_size    = app.project.grid_size
                app.canvas.snap.endpoint_tol = app.project.snap_tolerance
                app.canvas.fit_to_content()
                app._on_project_change()
            except Exception as e:
                print(f"[WARNING] No se pudo cargar '{path}': {e}", file=sys.stderr)

    app.run()


if __name__ == "__main__":
    main()
