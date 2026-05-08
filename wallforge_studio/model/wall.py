"""Modelo de muro: unidad fundamental de WallForge Studio."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class WallState(Enum):
    CONFIRMED = "confirmed"   # muro validado por el usuario
    DETECTED  = "detected"    # propuesto por detección automática, pendiente
    LOCKED    = "locked"      # bloqueado, no editable


@dataclass
class Wall:
    """Segmento de muro definido por dos extremos en metros (coord. mundo)."""

    p1:        Tuple[float, float]
    p2:        Tuple[float, float]
    thickness: float     = 0.15
    height:    float     = 2.5
    material:  str       = "wall_mat"
    state:     WallState = WallState.CONFIRMED
    id:        str       = field(default_factory=lambda: uuid.uuid4().hex[:8])

    # ── Geometría derivada ────────────────────────────────────────────────────

    def length(self) -> float:
        dx, dy = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        return math.hypot(dx, dy)

    def midpoint(self) -> Tuple[float, float]:
        return ((self.p1[0] + self.p2[0]) / 2.0,
                (self.p1[1] + self.p2[1]) / 2.0)

    def angle_rad(self) -> float:
        """Ángulo del segmento respecto al eje X, en radianes."""
        return math.atan2(self.p2[1] - self.p1[1], self.p2[0] - self.p1[0])

    def angle_deg(self) -> float:
        return math.degrees(self.angle_rad())

    def is_valid(self) -> bool:
        return self.length() > 1e-4 and self.thickness > 0.0 and self.height > 0.0

    def distance_to_point(self, x: float, y: float) -> float:
        """Distancia perpendicular del punto (x, y) al segmento (en metros)."""
        x1, y1 = self.p1
        x2, y2 = self.p2
        dx, dy  = x2 - x1, y2 - y1
        L2      = dx * dx + dy * dy
        if L2 < 1e-12:
            return math.hypot(x - x1, y - y1)
        t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
        return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))

    # ── Serialización ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id":        self.id,
            "p1":        list(self.p1),
            "p2":        list(self.p2),
            "thickness": self.thickness,
            "height":    self.height,
            "material":  self.material,
            "state":     self.state.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Wall":
        return cls(
            p1        = tuple(d["p1"]),
            p2        = tuple(d["p2"]),
            thickness = d.get("thickness", 0.15),
            height    = d.get("height", 2.5),
            material  = d.get("material", "wall_mat"),
            state     = WallState(d.get("state", "confirmed")),
            id        = d.get("id", uuid.uuid4().hex[:8]),
        )

    def copy(self) -> "Wall":
        return Wall.from_dict(self.to_dict())
