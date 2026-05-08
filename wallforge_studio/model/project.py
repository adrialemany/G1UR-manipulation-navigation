"""Estado global del proyecto WallForge."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .wall import Wall, WallState


@dataclass
class ExportSettings:
    wall_height:    float = 2.5
    wall_thickness: float = 0.15
    include_floor:  bool  = True
    include_lights: bool  = True
    include_robot:  bool  = False
    wall_prefix:    str   = "Wall"
    world_name:     str   = "wallforge_scene"
    robot_include:  str   = "g1_29dof.xml"


@dataclass
class BackgroundLayer:
    """Capa de fondo: imagen de plano arquitectónico."""
    image_path:   str
    ppm:          float                  = 100.0   # píxeles por metro
    origin_world: Tuple[float, float]    = (0.0, 0.0)  # centro imagen en coords mundo
    opacity:      float                  = 0.5
    visible:      bool                   = True
    locked:       bool                   = False

    # Datos en memoria — nunca serializados, se recargan al abrir el proyecto
    _np_bgr: Any = field(default=None, repr=False, compare=False)
    _pil_rgb: Any = field(default=None, repr=False, compare=False)

    @property
    def loaded(self) -> bool:
        return self._pil_rgb is not None

    def load_image(self) -> None:
        """Carga la imagen en memoria (numpy BGR + PIL RGB)."""
        from ..image.loader import load_image
        from PIL import Image
        import cv2
        bgr = load_image(Path(self.image_path))
        self._np_bgr = bgr
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._pil_rgb = Image.fromarray(rgb)

    @property
    def img_size(self) -> Tuple[int, int]:
        """(width, height) en píxeles o (0, 0) si no cargada."""
        if self._pil_rgb is None:
            return (0, 0)
        return self._pil_rgb.size

    def to_dict(self) -> dict:
        return {
            "image_path":   self.image_path,
            "ppm":          self.ppm,
            "origin_world": list(self.origin_world),
            "opacity":      self.opacity,
            "visible":      self.visible,
            "locked":       self.locked,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BackgroundLayer":
        return cls(
            image_path   = d["image_path"],
            ppm          = d.get("ppm", 100.0),
            origin_world = tuple(d.get("origin_world", [0.0, 0.0])),
            opacity      = d.get("opacity", 0.5),
            visible      = d.get("visible", True),
            locked       = d.get("locked", False),
        )


@dataclass
class Project:
    name:           str                      = "Untitled"
    walls:          List[Wall]               = field(default_factory=list)
    grid_size:      float                    = 0.5
    snap_enabled:   bool                     = True
    snap_tolerance: float                    = 0.3
    export:         ExportSettings           = field(default_factory=ExportSettings)
    background:     Optional[BackgroundLayer] = None
    file_path:      Optional[Path]           = None
    modified:       bool                     = False

    # ── Acceso a muros ────────────────────────────────────────────────────────

    def confirmed_walls(self) -> List[Wall]:
        return [w for w in self.walls if w.state != WallState.DETECTED]

    def add_wall(self, wall: Wall) -> None:
        self.walls.append(wall)
        self.modified = True

    def remove_wall(self, wall_id: str) -> bool:
        n = len(self.walls)
        self.walls = [w for w in self.walls if w.id != wall_id]
        removed = len(self.walls) < n
        if removed:
            self.modified = True
        return removed

    def get_wall(self, wall_id: str) -> Optional[Wall]:
        return next((w for w in self.walls if w.id == wall_id), None)

    def replace_wall(self, wall: Wall) -> None:
        for i, w in enumerate(self.walls):
            if w.id == wall.id:
                self.walls[i] = wall
                self.modified = True
                return

    # ── Serialización ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = {
            "name":           self.name,
            "grid_size":      self.grid_size,
            "snap_enabled":   self.snap_enabled,
            "snap_tolerance": self.snap_tolerance,
            "export": {
                "wall_height":    self.export.wall_height,
                "wall_thickness": self.export.wall_thickness,
                "include_floor":  self.export.include_floor,
                "include_lights": self.export.include_lights,
                "include_robot":  self.export.include_robot,
                "wall_prefix":    self.export.wall_prefix,
                "world_name":     self.export.world_name,
                "robot_include":  self.export.robot_include,
            },
            "walls": [w.to_dict() for w in self.walls],
        }
        if self.background is not None:
            d["background"] = self.background.to_dict()
        return d

    def save(self, path: Optional[Path] = None) -> Path:
        target = Path(path or self.file_path)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.file_path = target
        self.modified = False
        return target

    @classmethod
    def load(cls, path: Path) -> "Project":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        exp_d = d.get("export", {})
        valid_exp_fields = ExportSettings.__dataclass_fields__.keys()

        bg = None
        if "background" in d:
            bg = BackgroundLayer.from_dict(d["background"])
            if Path(bg.image_path).exists():
                try:
                    bg.load_image()
                except Exception:
                    pass  # imagen no disponible, se advierte al usuario

        proj = cls(
            name           = d.get("name", "Untitled"),
            grid_size      = d.get("grid_size", 0.5),
            snap_enabled   = d.get("snap_enabled", True),
            snap_tolerance = d.get("snap_tolerance", 0.3),
            export         = ExportSettings(
                **{k: v for k, v in exp_d.items() if k in valid_exp_fields}
            ),
            background = bg,
            file_path  = Path(path),
            walls      = [Wall.from_dict(wd) for wd in d.get("walls", [])],
        )
        return proj

    @classmethod
    def new(cls, name: str = "Untitled") -> "Project":
        return cls(name=name)
