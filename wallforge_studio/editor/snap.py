"""Motor de snapping para el canvas de WallForge Studio."""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING

from ..utils.geometry import snap_to_grid, closest_endpoint, snap_angle

if TYPE_CHECKING:
    from ..model.wall import Wall

Point = Tuple[float, float]


class SnapEngine:
    """
    Aplica snapping en orden de prioridad:
      1. Endpoint snap  (extremos de muros existentes)
      2. Grid snap      (rejilla configurable)
      3. Angle snap     (solo cuando shift está pulsado y hay punto ancla)
    """

    def __init__(self):
        self.grid_enabled:     bool  = True
        self.endpoint_enabled: bool  = True
        self.angle_step_deg:   float = 15.0
        self.grid_size:        float = 0.5
        self.endpoint_tol:     float = 0.3

    def apply(
        self,
        raw:       Point,
        walls:     "List[Wall]",
        anchor:    Optional[Point] = None,
        angle_snap: bool = False,
    ) -> Tuple[Point, str]:
        """
        Devuelve (punto_ajustado, hint) donde hint describe el tipo de snap activo.
        """
        # 1. Endpoint snap (máxima prioridad)
        if self.endpoint_enabled:
            ep = closest_endpoint(raw, walls, self.endpoint_tol)
            if ep is not None:
                return ep, "⊙ endpoint"

        # 2. Grid snap
        p = raw
        hint = ""
        if self.grid_enabled:
            p = snap_to_grid(raw, self.grid_size)
            hint = "⊞ grid"

        # 3. Angle snap (requiere anchor)
        if angle_snap and anchor is not None:
            p = snap_angle(anchor, p, self.angle_step_deg)
            hint = f"∠ {self.angle_step_deg:.0f}°"

        return p, hint
