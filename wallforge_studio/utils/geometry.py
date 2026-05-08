"""Utilidades geométricas para WallForge Studio."""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

Point = Tuple[float, float]


def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def midpoint(p1: Point, p2: Point) -> Point:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def angle_of_segment(p1: Point, p2: Point) -> float:
    """Ángulo del segmento en radianes respecto al eje X positivo."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def snap_to_grid(p: Point, grid: float) -> Point:
    if grid <= 0:
        return p
    return (round(p[0] / grid) * grid, round(p[1] / grid) * grid)


def snap_angle(p1: Point, p2: Point, step_deg: float = 15.0) -> Point:
    """Ajusta p2 al múltiplo de step_deg más cercano manteniendo la distancia."""
    a_deg = math.degrees(angle_of_segment(p1, p2))
    snapped = round(a_deg / step_deg) * step_deg
    L = distance(p1, p2)
    rad = math.radians(snapped)
    return (p1[0] + L * math.cos(rad), p1[1] + L * math.sin(rad))


def closest_endpoint(p: Point, walls, tolerance: float) -> Optional[Point]:
    """Devuelve el extremo de muro más cercano a p dentro de la tolerancia."""
    best_d, best_pt = tolerance, None
    for w in walls:
        for ep in (w.p1, w.p2):
            d = distance(p, ep)
            if d < best_d:
                best_d, best_pt = d, ep
    return best_pt


def world_extent(walls) -> float:
    """Radio máximo desde el origen para todos los muros (para <statistic extent>)."""
    if not walls:
        return 10.0
    pts = [p for w in walls for p in (w.p1, w.p2)]
    r = max(math.hypot(x, y) for x, y in pts)
    return max(10.0, math.ceil(r * 1.3))
