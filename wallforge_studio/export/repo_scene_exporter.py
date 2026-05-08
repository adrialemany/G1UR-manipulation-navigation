"""
Exportador MuJoCo XML adaptado al repositorio g1man.

Genera una escena compatible con el repo siguiendo scene_from_sdf_centered.xml
como plantilla estructural.

Diferencias clave respecto al exportador básico:
  - Incluye <include file="g1_29dof.xml"/> al inicio (robot + compiler radian)
  - Euler en RADIANES (porque g1_29dof.xml tiene <compiler angle="radian"/>)
  - <statistic center="0 0 1.0" extent="{N}"/> calculado del mapa
  - Paredes directamente en <worldbody>, sin body contenedor
  - Nombres: Wall_0, Wall_1, ... (capital W, sin ceros a la izquierda)
  - Luces a altura 10 (escala grande del mapa)
  - texrepeat="10 10" en groundplane
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import List

from ..model.wall import Wall, WallState
from ..model.project import ExportSettings
from ..utils.geometry import world_extent
from .mujoco_exporter import wall_to_geom, _floor, _lights_basic


# ══════════════════════════════════════════════════════════════════════════════
# Exportación adaptada al repo
# ══════════════════════════════════════════════════════════════════════════════

def export_repo_scene(walls: List[Wall], settings: ExportSettings, path: Path,
                      confirmed_only: bool = True) -> dict:
    """
    Genera un XML de escena compatible con el repositorio g1man.
    Usa radianes en euler (por <compiler angle="radian"/> del robot).
    Devuelve un resumen de exportación.
    """
    target = [w for w in walls if w.is_valid()
              and (not confirmed_only or w.state != WallState.DETECTED)]
    if not target:
        raise ValueError("No hay muros válidos para exportar.")

    extent = world_extent(target)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    world_name = settings.world_name or "wallforge_g1man_scene"

    lines: List[str] = [
        f'<!-- WallForge Studio — exportación para repo g1man — {ts} -->',
        f'<!-- Instalar en: mujoco/simulacion/{Path(path).name} -->',
        f'<!-- Ejecutar: python3 run_sim_ai_g1.py                -->',
        f'<mujoco model="{world_name}">',
        # El robot define <compiler angle="radian"/> → todos los ángulos son radianes
        f'  <include file="{settings.robot_include}"/>',
        "",
        f'  <statistic center="0 0 1.0" extent="{extent:.0f}"/>',
        "",
        "  <visual>",
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        '    <global azimuth="-130" elevation="-20"/>',
        "  </visual>",
        "",
        "  <asset>",
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"',
        '             width="512" height="3072"/>',
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge"',
        '             rgb1="0.22 0.22 0.22" rgb2="0.12 0.12 0.12"',
        '             markrgb="0.7 0.7 0.7" width="300" height="300"/>',
        '    <material name="groundplane" texture="groundplane" texuniform="true"',
        '              texrepeat="10 10" reflectance="0.15"/>',
        '    <material name="wall_mat" rgba="0.55 0.50 0.45 1" reflectance="0.05"/>',
        "  </asset>",
        "",
        "  <worldbody>",
        # Luces a altura 10 (escala de mapa grande, igual que referencia)
        '    <light pos="0 0 10" dir="0 0 -1" directional="true" diffuse="0.7 0.7 0.7"/>',
        '    <light pos="10 10 10" dir="0 0 -1" directional="true" diffuse="0.3 0.3 0.3"/>',
        '    <light pos="-10 -10 10" dir="0 0 -1" directional="true" diffuse="0.3 0.3 0.3"/>',
        "",
        '    <geom name="floor" type="plane" size="0 0 0.05"',
        '          material="groundplane" friction="2.5 2.5 2.5" group="3"/>',
        "",
        "    <!-- Paredes generadas por WallForge Studio -->",
    ]

    for i, w in enumerate(target):
        # Nombre sin ceros a la izquierda, capital W — igual que referencia
        name = f"Wall_{i}"
        lines.append(
            wall_to_geom(w, name, settings, angle_radians=True)
        )

    lines += [
        "  </worldbody>",
        "</mujoco>",
        "",
    ]

    Path(path).write_text("\n".join(lines), encoding="utf-8")

    return {
        "walls":      len(target),
        "file":       str(path),
        "world_name": world_name,
        "extent":     extent,
        "mode":       "repo_g1man",
        "robot":      settings.robot_include,
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "instructions": (
            f"Copia '{Path(path).name}' en mujoco/simulacion/ del repositorio g1man.\n"
            f"Luego ejecuta:  python3 mujoco/simulacion/run_sim_ai_g1.py\n"
            f"o usa el escenario directamente con MuJoCo Viewer."
        ),
    }
