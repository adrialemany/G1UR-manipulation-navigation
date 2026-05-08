"""Herramientas de edición del canvas de WallForge Studio."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .canvas import WallCanvas
    from ..model.wall import Wall

Point = Tuple[float, float]

# Radio en metros para detectar/mostrar handles de extremo
_EP_HANDLE_WORLD = 0.18   # tamaño visual mínimo en m
_EP_HANDLE_PX    = 14     # radio de detección mínimo en px


# ══════════════════════════════════════════════════════════════════════════════
# Clase base
# ══════════════════════════════════════════════════════════════════════════════

class BaseTool(ABC):
    name:        str = "base"
    cursor:      str = "crosshair"
    status_idle: str = ""

    def __init__(self, canvas: "WallCanvas"):
        self.canvas = canvas

    def on_press(self,   world: Point, shift: bool = False) -> None: ...
    def on_move(self,    world: Point, shift: bool = False) -> None: ...
    def on_release(self, world: Point, shift: bool = False) -> None: ...
    def on_escape(self) -> None: ...
    def draw_overlay(self) -> None: ...
    def activate(self) -> None: ...
    def deactivate(self) -> None:
        self.on_escape()


# ══════════════════════════════════════════════════════════════════════════════
# Herramienta: Seleccionar  (+ arrastrar extremos)
# ══════════════════════════════════════════════════════════════════════════════

class SelectTool(BaseTool):
    name        = "select"
    cursor      = "arrow"
    status_idle = (
        "Clic: seleccionar  |  Shift+clic: multi-selección  "
        "|  Arrastra un extremo (cuadrado naranja) para editarlo"
    )

    def __init__(self, canvas: "WallCanvas"):
        super().__init__(canvas)
        # estado de drag de extremo
        self._drag_wall_id:  Optional[str]   = None
        self._drag_endpoint: Optional[int]   = None   # 1 = p1, 2 = p2
        self._drag_orig_p1:  Optional[Point] = None   # backup pre-drag
        self._drag_orig_p2:  Optional[Point] = None
        self._drag_moved:    bool            = False
        # extremo sobre el que está el cursor (para overlay)
        self._hover_ep: Optional[Tuple[str, int]] = None

    # ── Utilidades ────────────────────────────────────────────────────────────

    def _ep_tol(self) -> float:
        """Tolerancia de detección de extremo en metros."""
        return max(_EP_HANDLE_WORLD, _EP_HANDLE_PX / self.canvas.zoom)

    def _endpoint_at(self, world: Point) -> Optional[Tuple[str, int]]:
        """Devuelve (wall_id, 1|2) del extremo seleccionado más cercano, o None."""
        c   = self.canvas
        tol = self._ep_tol()
        best: Optional[Tuple[str, int]] = None
        best_d = tol
        for wall in c.project.walls:
            if wall.id not in c.selected_ids:
                continue
            for idx, pt in enumerate((wall.p1, wall.p2), start=1):
                d = math.hypot(world[0] - pt[0], world[1] - pt[1])
                if d < best_d:
                    best_d, best = d, (wall.id, idx)
        return best

    # ── Eventos ───────────────────────────────────────────────────────────────

    def on_press(self, world: Point, shift: bool = False) -> None:
        c = self.canvas

        # ¿Clic sobre un extremo de muro seleccionado? → iniciar drag
        ep = self._endpoint_at(world)
        if ep:
            wall = c.project.get_wall(ep[0])
            if wall:
                self._drag_wall_id  = ep[0]
                self._drag_endpoint = ep[1]
                self._drag_orig_p1  = wall.p1
                self._drag_orig_p2  = wall.p2
                self._drag_moved    = False
                c.tk_canvas.config(cursor="fleur")
                c.status_var.set(
                    f"Arrastrando extremo {ep[1]} de «{ep[0][:6]}»  "
                    "|  Esc para cancelar"
                )
                return

        # Selección normal
        hit = c.wall_at(world)
        if hit:
            if shift:
                if hit.id in c.selected_ids:
                    c.selected_ids.discard(hit.id)
                else:
                    c.selected_ids.add(hit.id)
            else:
                c.selected_ids = {hit.id}
        else:
            if not shift:
                c.deselect_all()
        c.notify_change()
        c.refresh()

    def on_move(self, world: Point, shift: bool = False) -> None:
        c = self.canvas

        # ── En drag ──────────────────────────────────────────────────────────
        if self._drag_wall_id is not None:
            wall = c.project.get_wall(self._drag_wall_id)
            if wall is None:
                return
            # El snap usa como ancla el extremo opuesto (para angle-snap)
            anchor = wall.p2 if self._drag_endpoint == 1 else wall.p1
            snapped, hint = c.snap.apply(
                world, c.project.walls, anchor=anchor, angle_snap=shift
            )
            if self._drag_endpoint == 1:
                wall.p1 = snapped
            else:
                wall.p2 = snapped
            self._drag_moved    = True
            c.project.modified  = True
            length = math.hypot(wall.p2[0]-wall.p1[0], wall.p2[1]-wall.p1[1])
            c.status_var.set(
                f"Extremo: ({snapped[0]:.3f}, {snapped[1]:.3f}) m  "
                f"|  Long: {length:.3f} m  |  {hint}"
            )
            c.refresh()
            return

        # ── Hover normal ─────────────────────────────────────────────────────
        ep = self._endpoint_at(world)
        if ep != self._hover_ep:
            self._hover_ep = ep
            c.tk_canvas.config(cursor="fleur" if ep else self.cursor)
            c.refresh()
            return

        hit = c.wall_at(world)
        new_id = hit.id if hit else None
        if new_id != c.hover_id:
            c.hover_id = new_id
            c.refresh()

    def on_release(self, world: Point, shift: bool = False) -> None:
        c = self.canvas
        if self._drag_wall_id is not None and self._drag_moved:
            wall = c.project.get_wall(self._drag_wall_id)
            if wall:
                # Guardar undo desde el estado previo al drag
                cur_p1, cur_p2 = wall.p1, wall.p2
                wall.p1, wall.p2 = self._drag_orig_p1, self._drag_orig_p2
                c._push_undo()
                wall.p1, wall.p2 = cur_p1, cur_p2
            c.notify_change()
            c.refresh()
        self._reset_drag()
        c.tk_canvas.config(cursor=self.cursor)
        c.status_var.set(self.status_idle)

    def on_escape(self) -> None:
        c = self.canvas
        if self._drag_wall_id is not None:
            wall = c.project.get_wall(self._drag_wall_id)
            if wall and self._drag_orig_p1 and self._drag_orig_p2:
                wall.p1, wall.p2 = self._drag_orig_p1, self._drag_orig_p2
            self._reset_drag()
            c.tk_canvas.config(cursor=self.cursor)
            c.refresh()
            return
        c.deselect_all()
        c.refresh()

    def _reset_drag(self) -> None:
        self._drag_wall_id  = None
        self._drag_endpoint = None
        self._drag_orig_p1  = None
        self._drag_orig_p2  = None
        self._drag_moved    = False

    # ── Overlay ───────────────────────────────────────────────────────────────

    def draw_overlay(self) -> None:
        """Dibuja handles cuadrados en los extremos de los muros seleccionados."""
        c  = self.canvas
        tc = c.tk_canvas
        sz = max(5, int(c.zoom * _EP_HANDLE_WORLD))  # medio-lado del cuadrado

        for wall in c.project.walls:
            if wall.id not in c.selected_ids:
                continue
            for idx, pt in enumerate((wall.p1, wall.p2), start=1):
                sx, sy = c.w2s(*pt)
                is_hover = (self._hover_ep == (wall.id, idx))
                is_drag  = (self._drag_wall_id == wall.id
                            and self._drag_endpoint == idx)
                fill    = "#ffffff" if is_drag  else (
                          c.T["endpoint"] if is_hover else c.T["endpoint"])
                outline = "#ffffff" if (is_hover or is_drag) else "#888888"
                width   = 2 if (is_hover or is_drag) else 1
                tc.create_rectangle(
                    sx - sz, sy - sz, sx + sz, sy + sz,
                    fill=fill, outline=outline, width=width, tags="overlay"
                )


# ══════════════════════════════════════════════════════════════════════════════
# Herramienta: Dibujar muro
# ══════════════════════════════════════════════════════════════════════════════

class DrawTool(BaseTool):
    name        = "draw"
    cursor      = "crosshair"
    status_idle = "Clic para fijar el primer extremo del muro"

    def __init__(self, canvas: "WallCanvas"):
        super().__init__(canvas)
        self._p1:      Optional[Point] = None
        self._preview: Optional[Point] = None

    def on_press(self, world: Point, shift: bool = False) -> None:
        c = self.canvas
        snapped, hint = c.snap.apply(
            world, c.project.walls, anchor=self._p1, angle_snap=shift
        )
        if self._p1 is None:
            self._p1 = snapped
            c.status_var.set(
                f"Primer punto: ({snapped[0]:.2f}, {snapped[1]:.2f}) m"
                "  |  Clic para fijar extremo final  |  Esc para cancelar"
            )
        else:
            if snapped != self._p1:
                c._push_undo()
                from ..model.wall import Wall
                w = Wall(
                    p1        = self._p1,
                    p2        = snapped,
                    thickness = c.project.export.wall_thickness,
                    height    = c.project.export.wall_height,
                )
                c.project.add_wall(w)
                c.notify_change()
            self._reset()
        c.refresh()

    def on_move(self, world: Point, shift: bool = False) -> None:
        c = self.canvas
        if self._p1 is None:
            return
        snapped, hint = c.snap.apply(
            world, c.project.walls, anchor=self._p1, angle_snap=shift
        )
        self._preview = snapped
        c.refresh()
        from ..utils.geometry import distance
        d = distance(self._p1, snapped)
        c.status_var.set(f"Longitud: {d:.3f} m  |  {hint}")

    def on_escape(self) -> None:
        self._reset()
        self.canvas.refresh()

    def _reset(self) -> None:
        self._p1      = None
        self._preview = None
        self.canvas.status_var.set(self.status_idle)

    def draw_overlay(self) -> None:
        if self._p1 is None:
            return
        c  = self.canvas
        tc = c.tk_canvas
        sx1, sy1 = c.w2s(*self._p1)
        r = max(4, int(c.zoom * 0.07))

        # Punto ancla
        tc.create_oval(
            sx1 - r, sy1 - r, sx1 + r, sy1 + r,
            fill=c.T["endpoint"], outline="", tags="overlay"
        )
        # Línea de previsualización
        if self._preview:
            sx2, sy2 = c.w2s(*self._preview)
            tc.create_line(
                sx1, sy1, sx2, sy2,
                fill=c.T["preview"], width=2, dash=(8, 4), tags="overlay"
            )
            tc.create_oval(
                sx2 - r, sy2 - r, sx2 + r, sy2 + r,
                fill=c.T["preview"], outline="", tags="overlay"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Herramienta: Borrar
# ══════════════════════════════════════════════════════════════════════════════

class DeleteTool(BaseTool):
    name        = "delete"
    cursor      = "X_cursor"
    status_idle = "Clic sobre un muro para eliminarlo"

    def on_press(self, world: Point, shift: bool = False) -> None:
        c = self.canvas
        hit = c.wall_at(world)
        if hit:
            c._push_undo()
            c.project.remove_wall(hit.id)
            c.selected_ids.discard(hit.id)
            c.hover_id = None
            c.notify_change()
            c.refresh()

    def on_move(self, world: Point, shift: bool = False) -> None:
        c = self.canvas
        hit = c.wall_at(world)
        new_id = hit.id if hit else None
        if new_id != c.hover_id:
            c.hover_id = new_id
            c.refresh()

    def on_escape(self) -> None:
        self.canvas.hover_id = None
        self.canvas.refresh()
