"""Canvas 2D interactivo de WallForge Studio."""
from __future__ import annotations

import math
import tkinter as tk
from typing import Callable, Dict, List, Optional, Set, Tuple

from .snap import SnapEngine
from .tools import BaseTool, SelectTool, DrawTool, DeleteTool
from ..model.wall import Wall, WallState
from ..model.project import Project

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

Point = Tuple[float, float]

# ── Paleta visual ─────────────────────────────────────────────────────────────
T: Dict[str, str] = {
    "canvas_bg":      "#0d0d18",
    "grid":           "#1a1a2e",
    "grid_major":     "#252540",
    "grid_axis":      "#2e2e55",
    "wall_confirmed": "#89dceb",
    "wall_detected":  "#f9e2af",
    "wall_selected":  "#f38ba8",
    "wall_hover":     "#a6e3a1",
    "wall_locked":    "#585b70",
    "endpoint":       "#fab387",
    "preview":        "#cba6f7",
    "ruler_text":     "#4a4a6a",
    "origin_cross":   "#3a3a5a",
}

# Grosor de línea de muro en pantalla mínimo/máximo
_MIN_LW = 1
_MAX_LW = 8


class WallCanvas(tk.Frame):
    """
    Widget principal del editor.

    Sistema de coordenadas:
      - Mundo (metros): x→derecha, y→arriba  (igual que MuJoCo XY)
      - Pantalla (px):  x→derecha, y→abajo   (estándar)

    Transformaciones:
      sx = cx + (wx - pan_x) * zoom
      sy = cy - (wy - pan_y) * zoom
    donde (cx, cy) es el centro del canvas en píxeles.
    """

    T = T  # accessible from tools as canvas.T

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(self, parent: tk.Widget, project: Project,
                 status_var: tk.StringVar, **kw):
        super().__init__(parent, bg=T["canvas_bg"], **kw)

        self.project    = project
        self.status_var = status_var

        # Viewport
        self._zoom:  float = 60.0    # px/metro
        self._pan_x: float = 0.0    # metro del origen del mundo en x
        self._pan_y: float = 0.0

        # Estado de selección / hover
        self.selected_ids: Set[str]      = set()
        self.hover_id:     Optional[str] = None

        # Undo / Redo
        self._undo_stack: List[List[dict]] = []
        self._redo_stack: List[List[dict]] = []

        # Snap engine
        self.snap = SnapEngine()
        self.snap.grid_size   = project.grid_size
        self.snap.endpoint_tol = project.snap_tolerance

        # Herramientas
        self._tools: Dict[str, BaseTool] = {
            "select": SelectTool(self),
            "draw":   DrawTool(self),
            "delete": DeleteTool(self),
        }
        self._active: BaseTool = self._tools["draw"]

        # Pan con botón central/derecho
        self._pan_start:  Optional[Tuple[int, int]]      = None
        self._pan_origin: Optional[Tuple[float, float]]  = None

        # Callbacks de cambio (actualizar propiedades, título, etc.)
        self._on_change: List[Callable] = []

        # Widget Tk
        self.tk_canvas = tk.Canvas(
            self, bg=T["canvas_bg"],
            highlightthickness=0, cursor="crosshair",
        )
        self.tk_canvas.pack(fill=tk.BOTH, expand=True)

        self._bind_events()
        self.after(80, self._center_origin)

    # ── Coordenadas ───────────────────────────────────────────────────────────

    @property
    def zoom(self) -> float:
        return self._zoom

    def w2s(self, wx: float, wy: float) -> Tuple[int, int]:
        cx, cy = self._canvas_center()
        return (
            int(cx + (wx - self._pan_x) * self._zoom),
            int(cy - (wy - self._pan_y) * self._zoom),
        )

    def s2w(self, sx: int, sy: int) -> Tuple[float, float]:
        cx, cy = self._canvas_center()
        return (
            (sx - cx) / self._zoom + self._pan_x,
            -(sy - cy) / self._zoom + self._pan_y,
        )

    def _canvas_center(self) -> Tuple[float, float]:
        w = self.tk_canvas.winfo_width()  or 800
        h = self.tk_canvas.winfo_height() or 600
        return w / 2.0, h / 2.0

    def _center_origin(self) -> None:
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.refresh()

    # ── Zoom / Pan ────────────────────────────────────────────────────────────

    def _on_scroll(self, event: tk.Event) -> None:
        up = getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4
        factor = 1.15 if up else 1.0 / 1.15
        wx, wy = self.s2w(event.x, event.y)
        self._zoom = max(4.0, min(600.0, self._zoom * factor))
        cx, cy = self._canvas_center()
        self._pan_x = wx - (event.x - cx) / self._zoom
        self._pan_y = wy + (event.y - cy) / self._zoom
        self.refresh()

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start  = (event.x, event.y)
        self._pan_origin = (self._pan_x, self._pan_y)
        self.tk_canvas.config(cursor="fleur")

    def _on_pan_move(self, event: tk.Event) -> None:
        if self._pan_start is None:
            return
        dx = (event.x - self._pan_start[0]) / self._zoom
        dy = (event.y - self._pan_start[1]) / self._zoom
        self._pan_x = self._pan_origin[0] - dx
        self._pan_y = self._pan_origin[1] + dy
        self.refresh()

    def _on_pan_end(self, event: tk.Event) -> None:
        self._pan_start  = None
        self._pan_origin = None
        self.tk_canvas.config(cursor=self._active.cursor)

    # ── Eventos de herramienta ────────────────────────────────────────────────

    def _on_lpress(self, event: tk.Event) -> None:
        world = self.s2w(event.x, event.y)
        shift = bool(event.state & 0x1)
        self._active.on_press(world, shift)

    def _on_lrelease(self, event: tk.Event) -> None:
        world = self.s2w(event.x, event.y)
        shift = bool(event.state & 0x1)
        self._active.on_release(world, shift)

    def _on_motion(self, event: tk.Event) -> None:
        world = self.s2w(event.x, event.y)
        shift = bool(event.state & 0x1)
        self._active.on_move(world, shift)
        # Coordenadas en status bar
        if self.status_var.get().startswith("(") or not self.status_var.get():
            self.status_var.set(f"({world[0]:.3f}, {world[1]:.3f}) m")

    def _on_escape(self, event: tk.Event) -> None:
        self._active.on_escape()

    # ── Binding ───────────────────────────────────────────────────────────────

    def _bind_events(self) -> None:
        c = self.tk_canvas
        c.bind("<MouseWheel>",      self._on_scroll)
        c.bind("<Button-4>",        self._on_scroll)
        c.bind("<Button-5>",        self._on_scroll)
        c.bind("<ButtonPress-2>",   self._on_pan_start)
        c.bind("<B2-Motion>",       self._on_pan_move)
        c.bind("<ButtonRelease-2>", self._on_pan_end)
        c.bind("<ButtonPress-3>",   self._on_pan_start)
        c.bind("<B3-Motion>",       self._on_pan_move)
        c.bind("<ButtonRelease-3>", self._on_pan_end)
        c.bind("<ButtonPress-1>",   self._on_lpress)
        c.bind("<ButtonRelease-1>", self._on_lrelease)
        c.bind("<Motion>",          self._on_motion)
        c.bind("<Configure>",       lambda e: self.refresh())
        root = self.winfo_toplevel()
        root.bind("<Escape>", self._on_escape)

    # ── API de herramientas ───────────────────────────────────────────────────

    def set_tool(self, name: str) -> None:
        self._active.deactivate()
        self._active = self._tools[name]
        self._active.activate()
        self.tk_canvas.config(cursor=self._active.cursor)
        self.hover_id = None
        self.status_var.set(self._active.status_idle)
        self.refresh()

    @property
    def active_tool_name(self) -> str:
        return self._active.name

    # ── Selección ────────────────────────────────────────────────────────────

    def deselect_all(self) -> None:
        self.selected_ids.clear()

    def wall_at(self, world: Point) -> Optional[Wall]:
        """Muro más cercano al punto dado; tolerancia adaptada al zoom."""
        tol = max(0.15, 12.0 / self._zoom)
        best_w, best_d = None, tol
        for w in self.project.walls:
            d = w.distance_to_point(*world)
            if d < best_d:
                best_d, best_w = d, w
        return best_w

    def delete_selected(self) -> None:
        if not self.selected_ids:
            return
        self._push_undo()
        for wid in list(self.selected_ids):
            self.project.remove_wall(wid)
        self.selected_ids.clear()
        self.notify_change()
        self.refresh()

    # ── Undo / Redo ───────────────────────────────────────────────────────────

    def _push_undo(self) -> None:
        self._undo_stack.append([w.to_dict() for w in self.project.walls])
        self._redo_stack.clear()
        if len(self._undo_stack) > 60:
            self._undo_stack.pop(0)

    def undo(self) -> None:
        if not self._undo_stack:
            return
        self._redo_stack.append([w.to_dict() for w in self.project.walls])
        self.project.walls = [Wall.from_dict(d) for d in self._undo_stack.pop()]
        self.project.modified = True
        self.selected_ids.clear()
        self.notify_change()
        self.refresh()

    def redo(self) -> None:
        if not self._redo_stack:
            return
        self._undo_stack.append([w.to_dict() for w in self.project.walls])
        self.project.walls = [Wall.from_dict(d) for d in self._redo_stack.pop()]
        self.project.modified = True
        self.selected_ids.clear()
        self.notify_change()
        self.refresh()

    def can_undo(self) -> bool: return bool(self._undo_stack)
    def can_redo(self) -> bool: return bool(self._redo_stack)

    # ── Callbacks ────────────────────────────────────────────────────────────

    def on_change(self, cb: Callable) -> None:
        self._on_change.append(cb)

    def notify_change(self) -> None:
        for cb in self._on_change:
            cb()

    # ── Renderizado ───────────────────────────────────────────────────────────

    def refresh(self) -> None:
        c = self.tk_canvas
        c.delete("all")
        self._draw_grid()
        self._draw_background()
        self._draw_walls()
        self._active.draw_overlay()

    def _draw_background(self) -> None:
        if not _HAS_PIL:
            return
        bg = self.project.background
        if bg is None or not bg.visible or not bg.loaded:
            return

        img_w, img_h = bg.img_size
        if img_w == 0 or img_h == 0:
            return

        ppm    = bg.ppm
        ox, oy = bg.origin_world

        # Esquinas de la imagen en coordenadas mundo (y↑)
        left_w  = ox - img_w / (2.0 * ppm)
        top_w   = oy + img_h / (2.0 * ppm)   # mayor y → menor sy
        right_w = ox + img_w / (2.0 * ppm)
        bot_w   = oy - img_h / (2.0 * ppm)

        sx_left,  sy_top = self.w2s(left_w,  top_w)
        sx_right, sy_bot = self.w2s(right_w, bot_w)

        cw = self.tk_canvas.winfo_width()  or 800
        ch = self.tk_canvas.winfo_height() or 600

        # Recorte al viewport
        dst_x0 = max(0, sx_left);  dst_y0 = max(0, sy_top)
        dst_x1 = min(cw, sx_right); dst_y1 = min(ch, sy_bot)
        dst_w  = dst_x1 - dst_x0
        dst_h  = dst_y1 - dst_y0
        if dst_w <= 0 or dst_h <= 0:
            return

        # Fracción de la imagen a mostrar
        span_sx = max(sx_right - sx_left, 1)
        span_sy = max(sy_bot   - sy_top,  1)
        px0 = int((dst_x0 - sx_left) / span_sx * img_w)
        py0 = int((dst_y0 - sy_top)  / span_sy * img_h)
        px1 = int((dst_x1 - sx_left) / span_sx * img_w)
        py1 = int((dst_y1 - sy_top)  / span_sy * img_h)
        px0 = max(0, px0); py0 = max(0, py0)
        px1 = min(img_w, px1); py1 = min(img_h, py1)
        if px1 <= px0 or py1 <= py0:
            return

        crop = bg._pil_rgb.crop((px0, py0, px1, py1))
        resample = Image.NEAREST if self._zoom >= 1.0 else Image.LANCZOS
        crop = crop.resize((dst_w, dst_h), resample)

        # Mezcla con color de fondo del canvas para la opacidad
        if bg.opacity < 0.99:
            canvas_color = (13, 13, 24)   # #0d0d18
            bg_layer = Image.new("RGB", (dst_w, dst_h), canvas_color)
            crop = Image.blend(bg_layer, crop.convert("RGB"), bg.opacity)

        self._bg_photo = ImageTk.PhotoImage(crop.convert("RGB"))
        self.tk_canvas.create_image(
            dst_x0, dst_y0, image=self._bg_photo, anchor="nw", tags="background"
        )

    def _draw_grid(self) -> None:
        c  = self.tk_canvas
        cw = c.winfo_width()  or 800
        ch = c.winfo_height() or 600
        gs = self.project.grid_size

        x0, y0 = self.s2w(0,  0)
        x1, y1 = self.s2w(cw, ch)
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0

        major = gs * 5

        # Líneas verticales
        ix = math.floor(x0 / gs) * gs
        while ix <= x1 + gs:
            sx, _ = self.w2s(ix, 0)
            is_axis  = abs(ix) < gs * 0.05
            is_major = not is_axis and abs(round(ix / major) * major - ix) < gs * 0.05
            color = (T["grid_axis"] if is_axis
                     else T["grid_major"] if is_major
                     else T["grid"])
            c.create_line(sx, 0, sx, ch, fill=color, tags="grid")
            if is_major or is_axis:
                _, sy_lbl = self.w2s(0, 0)
                sy_lbl = max(12, min(ch - 12, sy_lbl + 14))
                c.create_text(
                    sx + 3, sy_lbl,
                    text=f"{ix:.0f}m", fill=T["ruler_text"],
                    font=("Helvetica", 7), anchor="w", tags="grid"
                )
            ix = round(ix + gs, 9)

        # Líneas horizontales
        iy = math.floor(y0 / gs) * gs
        while iy <= y1 + gs:
            _, sy = self.w2s(0, iy)
            is_axis  = abs(iy) < gs * 0.05
            is_major = not is_axis and abs(round(iy / major) * major - iy) < gs * 0.05
            color = (T["grid_axis"] if is_axis
                     else T["grid_major"] if is_major
                     else T["grid"])
            c.create_line(0, sy, cw, sy, fill=color, tags="grid")
            if (is_major or is_axis) and abs(iy) > gs * 0.05:
                sx_lbl, _ = self.w2s(0, 0)
                sx_lbl = max(12, min(cw - 30, sx_lbl + 4))
                c.create_text(
                    sx_lbl, sy - 8,
                    text=f"{iy:.0f}m", fill=T["ruler_text"],
                    font=("Helvetica", 7), anchor="w", tags="grid"
                )
            iy = round(iy + gs, 9)

    def _draw_walls(self) -> None:
        c  = self.tk_canvas
        lw_base = max(_MIN_LW, min(_MAX_LW, int(self._zoom * self.project.export.wall_thickness / 3)))

        for wall in self.project.walls:
            sx1, sy1 = self.w2s(*wall.p1)
            sx2, sy2 = self.w2s(*wall.p2)

            # Color y grosor según estado
            if wall.id in self.selected_ids:
                color, lw = T["wall_selected"], lw_base + 2
            elif wall.id == self.hover_id:
                color, lw = T["wall_hover"], lw_base + 1
            elif wall.state == WallState.DETECTED:
                color, lw = T["wall_detected"], lw_base
            elif wall.state == WallState.LOCKED:
                color, lw = T["wall_locked"], lw_base
            else:
                color, lw = T["wall_confirmed"], lw_base

            c.create_line(
                sx1, sy1, sx2, sy2,
                fill=color, width=lw,
                capstyle=tk.ROUND, tags="wall"
            )

            # Puntos extremo (pequeños círculos)
            r = max(2, int(self._zoom * 0.05))
            for sx, sy in ((sx1, sy1), (sx2, sy2)):
                c.create_oval(
                    sx - r, sy - r, sx + r, sy + r,
                    fill=color, outline="", tags="wall"
                )

    # ── Reset de vista ────────────────────────────────────────────────────────

    def fit_to_content(self) -> None:
        """Ajusta zoom y pan para ver todos los muros."""
        if not self.project.walls:
            self._center_origin()
            return
        pts = [p for w in self.project.walls for p in (w.p1, w.p2)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        self._pan_x, self._pan_y = mx, my
        cw = self.tk_canvas.winfo_width()  or 800
        ch = self.tk_canvas.winfo_height() or 600
        span_x = max(xs) - min(xs) or 10.0
        span_y = max(ys) - min(ys) or 10.0
        margin = 0.85
        self._zoom = min(cw / span_x, ch / span_y) * margin
        self.refresh()
