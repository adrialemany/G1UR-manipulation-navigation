"""
Exportador MuJoCo XML básico (ángulos en grados, sin robot).

Regla de conversión  segmento → geom box:
  p1=(x1,y1), p2=(x2,y2) en metros (coord. mundo del editor)
  → pos  = (midx, midy, height/2)                [centro del geom]
  → size = (length/2, thickness/2, height/2)     [half-extents MuJoCo]
  → euler = "0 0 {angle_deg:.4f}"               [grados — default MuJoCo]
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import List

from ..model.wall import Wall, WallState
from ..model.project import ExportSettings
from ..utils.geometry import world_extent


# ══════════════════════════════════════════════════════════════════════════════
# Conversión individual
# ══════════════════════════════════════════════════════════════════════════════

def wall_to_geom(wall: Wall, name: str, settings: ExportSettings,
                 angle_radians: bool = False) -> str:
    """Devuelve la línea XML <geom …/> para un muro."""
    if not wall.is_valid():
        raise ValueError(f"Muro inválido: id={wall.id}")
    mx, my  = wall.midpoint()
    L       = wall.length()
    h       = wall.height
    t       = wall.thickness
    angle   = wall.angle_rad()
    euler_z = f"{angle:.6f}" if angle_radians else f"{math.degrees(angle):.4f}"
    return (
        f'    <geom name="{name}" type="box"'
        f' pos="{mx:.4f} {my:.4f} {h / 2:.4f}"'
        f' size="{L / 2:.4f} {t / 2:.4f} {h / 2:.4f}"'
        f' euler="0 0 {euler_z}"'
        f' material="{wall.material}"'
        f' friction="1 0.05 0.01" group="3"/>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# Exportación básica (escena genérica, sin robot, ángulos en grados)
# ══════════════════════════════════════════════════════════════════════════════

def export_basic(walls: List[Wall], settings: ExportSettings, path: Path,
                 confirmed_only: bool = True) -> dict:
    """
    Genera un XML MuJoCo básico y lo guarda en path.
    Devuelve un resumen de exportación.
    """
    target = [w for w in walls if w.is_valid()
              and (not confirmed_only or w.state != WallState.DETECTED)]
    if not target:
        raise ValueError("No hay muros válidos para exportar.")

    prefix = settings.wall_prefix
    lines  = _scene_header(settings.world_name, settings)
    lines += ["  <worldbody>"]

    if settings.include_lights:
        lines += _lights_basic()
    if settings.include_floor:
        lines.append(_floor())

    for i, w in enumerate(target):
        lines.append(wall_to_geom(w, f"{prefix}_{i}", settings, angle_radians=False))

    lines += ["  </worldbody>", "</mujoco>", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")

    return {
        "walls":      len(target),
        "file":       str(path),
        "world_name": settings.world_name,
        "mode":       "basic",
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers de fragmentos XML
# ══════════════════════════════════════════════════════════════════════════════

def _scene_header(world_name: str, settings: ExportSettings) -> List[str]:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return [
        f'<!-- WallForge Studio — {ts} -->',
        f'<mujoco model="{world_name}">',
        "  <visual>",
        '    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        '    <global azimuth="-130" elevation="-20"/>',
        "  </visual>",
        "  <asset>",
        '    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"',
        '             rgb2="0 0 0" width="512" height="3072"/>',
        '    <texture type="2d" name="groundplane" builtin="checker" mark="edge"',
        '             rgb1="0.22 0.22 0.22" rgb2="0.12 0.12 0.12"',
        '             markrgb="0.7 0.7 0.7" width="300" height="300"/>',
        '    <material name="groundplane" texture="groundplane" texuniform="true"',
        '              texrepeat="10 10" reflectance="0.15"/>',
        '    <material name="wall_mat" rgba="0.55 0.50 0.45 1" reflectance="0.05"/>',
        "  </asset>",
    ]


def _lights_basic() -> List[str]:
    return [
        '    <light pos="0 0 10" dir="0 0 -1" directional="true" diffuse="0.7 0.7 0.7"/>',
        '    <light pos="10 10 10" dir="0 0 -1" directional="true" diffuse="0.3 0.3 0.3"/>',
        '    <light pos="-10 -10 10" dir="0 0 -1" directional="true" diffuse="0.3 0.3 0.3"/>',
    ]


def _floor() -> str:
    return (
        '    <geom name="floor" type="plane" size="0 0 0.05"'
        ' material="groundplane" friction="2.5 2.5 2.5" group="3"/>'
    )
