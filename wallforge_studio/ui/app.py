"""Ventana principal de WallForge Studio."""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
from typing import Optional

from ..model.project import Project
from ..model.wall import Wall, WallState
from ..editor.canvas import WallCanvas
from ..export.mujoco_exporter import export_basic
from ..export.repo_scene_exporter import export_repo_scene

# ── Paleta ────────────────────────────────────────────────────────────────────
C = {
    "bg":       "#1e1e2e",
    "bg_panel": "#252535",
    "bg_input": "#313145",
    "text":     "#cdd6f4",
    "text_dim": "#7f849c",
    "accent":   "#89b4fa",
    "btn":      "#313145",
    "btn_act":  "#45475a",
    "sep":      "#383850",
    "danger":   "#f38ba8",
    "success":  "#a6e3a1",
}

_FONT      = ("Helvetica", 10)
_FONT_BOLD = ("Helvetica", 10, "bold")
_FONT_SM   = ("Helvetica", 8)
_FONT_MONO = ("Courier", 9)

APP_TITLE = "WallForge Studio"
VERSION   = "1.0.0-phase1"


# ══════════════════════════════════════════════════════════════════════════════
# Aplicación principal
# ══════════════════════════════════════════════════════════════════════════════

class WallForgeApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        self.root.geometry("1280x760")
        self.root.minsize(900, 600)
        self.root.configure(bg=C["bg"])

        self.project: Project = Project.new("Untitled")
        self.status_var = tk.StringVar(value="Listo")

        self._build_ui()
        self._bind_shortcuts()
        self._update_title()

        # Notificar canvas cuando cambie el proyecto
        self.canvas.on_change(self._on_project_change)

    # ── Construcción UI ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_menu()

        # Marco principal (toolbar | canvas | properties)
        main = tk.Frame(self.root, bg=C["bg"])
        main.pack(fill=tk.BOTH, expand=True)

        self._toolbar_frame = tk.Frame(main, bg=C["bg_panel"], width=72)
        self._toolbar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self._toolbar_frame.pack_propagate(False)

        self._props_frame = tk.Frame(main, bg=C["bg_panel"], width=230)
        self._props_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self._props_frame.pack_propagate(False)

        canvas_frame = tk.Frame(main, bg="#0d0d18")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas
        self.canvas = WallCanvas(canvas_frame, self.project, self.status_var)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._build_toolbar()
        self._build_properties()
        self._build_statusbar()
        self._update_tool_buttons()  # después de crear _tool_label en statusbar

    def _build_menu(self) -> None:
        mb = tk.Menu(self.root, bg=C["bg_panel"], fg=C["text"],
                     activebackground=C["btn_act"], activeforeground=C["text"],
                     relief=tk.FLAT, bd=0)
        self.root.config(menu=mb)

        # File
        file_m = tk.Menu(mb, tearoff=0, bg=C["bg_panel"], fg=C["text"],
                         activebackground=C["btn_act"], activeforeground=C["text"])
        file_m.add_command(label="Nuevo              Ctrl+N", command=self._new_project)
        file_m.add_command(label="Abrir…             Ctrl+O", command=self._open_project)
        file_m.add_separator()
        file_m.add_command(label="Guardar            Ctrl+S", command=self._save_project)
        file_m.add_command(label="Guardar como…   Ctrl+Shift+S", command=self._save_as)
        file_m.add_separator()
        file_m.add_command(label="Salir              Ctrl+Q", command=self._quit)
        mb.add_cascade(label="Archivo", menu=file_m)

        # Edit
        edit_m = tk.Menu(mb, tearoff=0, bg=C["bg_panel"], fg=C["text"],
                         activebackground=C["btn_act"], activeforeground=C["text"])
        edit_m.add_command(label="Deshacer           Ctrl+Z", command=self._undo)
        edit_m.add_command(label="Rehacer            Ctrl+Y", command=self._redo)
        edit_m.add_separator()
        edit_m.add_command(label="Eliminar selección   Del", command=self._delete_selected)
        edit_m.add_command(label="Seleccionar todo   Ctrl+A", command=self._select_all)
        edit_m.add_command(label="Deseleccionar      Esc",    command=self._deselect)
        mb.add_cascade(label="Editar", menu=edit_m)

        # View
        view_m = tk.Menu(mb, tearoff=0, bg=C["bg_panel"], fg=C["text"],
                         activebackground=C["btn_act"], activeforeground=C["text"])
        view_m.add_command(label="Ajustar vista al contenido   F", command=self._fit)
        view_m.add_command(label="Centrar origen               Home", command=self._center)
        view_m.add_separator()
        self._grid_var = tk.BooleanVar(value=True)
        self._snap_var = tk.BooleanVar(value=True)
        view_m.add_checkbutton(label="Rejilla    G", variable=self._grid_var,
                               command=self._toggle_grid)
        view_m.add_checkbutton(label="Snap       Ctrl+G", variable=self._snap_var,
                               command=self._toggle_snap)
        mb.add_cascade(label="Vista", menu=view_m)

        # Tools
        tool_m = tk.Menu(mb, tearoff=0, bg=C["bg_panel"], fg=C["text"],
                         activebackground=C["btn_act"], activeforeground=C["text"])
        tool_m.add_command(label="Seleccionar   S", command=lambda: self._set_tool("select"))
        tool_m.add_command(label="Dibujar       D", command=lambda: self._set_tool("draw"))
        tool_m.add_command(label="Borrar        X", command=lambda: self._set_tool("delete"))
        mb.add_cascade(label="Herramientas", menu=tool_m)

        # Plano de fondo
        plano_m = tk.Menu(mb, tearoff=0, bg=C["bg_panel"], fg=C["text"],
                          activebackground=C["btn_act"], activeforeground=C["text"])
        plano_m.add_command(label="Importar imagen / PDF…   Ctrl+I", command=self._import_background)
        plano_m.add_command(label="Calibrar escala (px/m)…", command=self._calibrate_background)
        plano_m.add_separator()
        self._bg_visible_var = tk.BooleanVar(value=True)
        plano_m.add_checkbutton(label="Mostrar plano", variable=self._bg_visible_var,
                                command=self._toggle_bg_visible)
        self._bg_locked_var = tk.BooleanVar(value=False)
        plano_m.add_checkbutton(label="Bloquear plano", variable=self._bg_locked_var,
                                command=self._toggle_bg_locked)
        plano_m.add_separator()
        plano_m.add_command(label="Detectar paredes del plano…   Ctrl+D", command=self._open_detection_dialog)
        plano_m.add_command(label="Confirmar muros detectados", command=self._confirm_detected)
        plano_m.add_separator()
        plano_m.add_command(label="Eliminar plano de fondo", command=self._remove_background)
        mb.add_cascade(label="Plano", menu=plano_m)

        # Export
        exp_m = tk.Menu(mb, tearoff=0, bg=C["bg_panel"], fg=C["text"],
                        activebackground=C["btn_act"], activeforeground=C["text"])
        exp_m.add_command(label="Exportar MuJoCo XML básico…", command=self._export_basic)
        exp_m.add_command(label="Exportar para repo g1man…",   command=self._export_repo)
        mb.add_cascade(label="Exportar", menu=exp_m)

        # Help
        help_m = tk.Menu(mb, tearoff=0, bg=C["bg_panel"], fg=C["text"],
                         activebackground=C["btn_act"], activeforeground=C["text"])
        help_m.add_command(label="Atajos de teclado", command=self._show_shortcuts)
        help_m.add_separator()
        help_m.add_command(label=f"Acerca de {APP_TITLE}", command=self._show_about)
        mb.add_cascade(label="Ayuda", menu=help_m)

    def _build_toolbar(self) -> None:
        f = self._toolbar_frame
        self._tool_btns: dict = {}

        def section(text: str) -> None:
            tk.Label(f, text=text, bg=C["bg_panel"], fg=C["text_dim"],
                     font=_FONT_SM).pack(pady=(10, 2))
            tk.Frame(f, bg=C["sep"], height=1).pack(fill=tk.X, padx=6)

        def tool_btn(label: str, key: str, tool: str) -> None:
            b = tk.Button(
                f, text=f"{label}\n[{key}]",
                bg=C["btn"], fg=C["text"], font=_FONT_SM,
                relief=tk.FLAT, padx=4, pady=6, width=7,
                activebackground=C["accent"], activeforeground=C["bg"],
                command=lambda t=tool: self._set_tool(t),
            )
            b.pack(pady=2, padx=6, fill=tk.X)
            self._tool_btns[tool] = b

        def action_btn(label: str, key: str, cmd) -> None:
            tk.Button(
                f, text=f"{label}\n[{key}]",
                bg=C["btn"], fg=C["text"], font=_FONT_SM,
                relief=tk.FLAT, padx=4, pady=6, width=7,
                activebackground=C["btn_act"], activeforeground=C["text"],
                command=cmd,
            ).pack(pady=2, padx=6, fill=tk.X)

        section("HERR.")
        tool_btn("Selec.", "S", "select")
        tool_btn("Dibujar", "D", "draw")
        tool_btn("Borrar", "X", "delete")

        section("EDITAR")
        action_btn("Deshacer", "^Z", self._undo)
        action_btn("Rehacer",  "^Y", self._redo)
        action_btn("Eliminar", "Del", self._delete_selected)

        section("VISTA")
        action_btn("Ajustar", "F",    self._fit)
        action_btn("Centro",  "Home", self._center)

        section("EXPORT")
        tk.Button(
            f, text="MuJoCo\nBásico",
            bg="#1e3a1e", fg=C["success"], font=_FONT_SM,
            relief=tk.FLAT, padx=4, pady=6, width=7,
            activebackground=C["success"], activeforeground=C["bg"],
            command=self._export_basic,
        ).pack(pady=2, padx=6, fill=tk.X)

        tk.Button(
            f, text="Repo\ng1man",
            bg="#1e2a3e", fg=C["accent"], font=_FONT_SM,
            relief=tk.FLAT, padx=4, pady=6, width=7,
            activebackground=C["accent"], activeforeground=C["bg"],
            command=self._export_repo,
        ).pack(pady=2, padx=6, fill=tk.X)


    def _build_properties(self) -> None:
        f = self._props_frame

        tk.Label(f, text="PROPIEDADES", bg=C["bg_panel"], fg=C["text_dim"],
                 font=_FONT_SM).pack(pady=(12, 2), padx=10, anchor="w")
        tk.Frame(f, bg=C["sep"], height=1).pack(fill=tk.X, padx=8, pady=2)

        # Panel de muro seleccionado
        self._prop_frame = tk.Frame(f, bg=C["bg_panel"])
        self._prop_frame.pack(fill=tk.X, padx=8, pady=6)

        self._prop_labels: dict = {}
        fields = [
            ("ID",        "id",        False),
            ("Longitud",  "length",    False),
            ("Ángulo",    "angle",     False),
            ("Grosor (m)", "thickness", True),
            ("Altura (m)", "height",   True),
            ("Material",  "material",  True),
            ("Estado",    "state",     False),
        ]
        for label, key, editable in fields:
            row = tk.Frame(self._prop_frame, bg=C["bg_panel"])
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label + ":", bg=C["bg_panel"], fg=C["text_dim"],
                     font=_FONT_SM, width=11, anchor="w").pack(side=tk.LEFT)
            if editable:
                var = tk.StringVar()
                e = tk.Entry(row, textvariable=var, bg=C["bg_input"], fg=C["text"],
                             insertbackground=C["text"], relief=tk.FLAT,
                             font=_FONT_MONO, width=10)
                e.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self._prop_labels[key] = (var, e)
            else:
                lbl = tk.Label(row, text="—", bg=C["bg_panel"], fg=C["text"],
                               font=_FONT_MONO, anchor="w")
                lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self._prop_labels[key] = lbl

        # Botón aplicar
        self._apply_btn = tk.Button(
            self._prop_frame, text="Aplicar cambios",
            bg=C["btn"], fg=C["text"], font=_FONT_SM, relief=tk.FLAT,
            padx=4, pady=4,
            activebackground=C["accent"], activeforeground=C["bg"],
            command=self._apply_properties,
        )
        self._apply_btn.pack(fill=tk.X, pady=(6, 2))

        # Sección info del proyecto
        tk.Frame(f, bg=C["sep"], height=1).pack(fill=tk.X, padx=8, pady=(12, 2))
        tk.Label(f, text="PROYECTO", bg=C["bg_panel"], fg=C["text_dim"],
                 font=_FONT_SM).pack(pady=(2, 2), padx=10, anchor="w")

        self._info_walls = tk.Label(f, text="Muros: 0", bg=C["bg_panel"],
                                    fg=C["text"], font=_FONT_SM)
        self._info_walls.pack(padx=10, anchor="w")
        self._info_sel = tk.Label(f, text="Seleccionados: 0", bg=C["bg_panel"],
                                  fg=C["text"], font=_FONT_SM)
        self._info_sel.pack(padx=10, anchor="w")

        # Configuración de snap y rejilla
        tk.Frame(f, bg=C["sep"], height=1).pack(fill=tk.X, padx=8, pady=(12, 2))
        tk.Label(f, text="CONFIGURACIÓN", bg=C["bg_panel"], fg=C["text_dim"],
                 font=_FONT_SM).pack(pady=(2, 4), padx=10, anchor="w")

        def cfg_row(label: str, var: tk.Variable) -> None:
            row = tk.Frame(f, bg=C["bg_panel"])
            row.pack(fill=tk.X, padx=10, pady=1)
            tk.Label(row, text=label, bg=C["bg_panel"], fg=C["text"],
                     font=_FONT_SM, width=14, anchor="w").pack(side=tk.LEFT)
            tk.Entry(row, textvariable=var, bg=C["bg_input"], fg=C["text"],
                     insertbackground=C["text"], relief=tk.FLAT,
                     font=_FONT_MONO, width=6).pack(side=tk.LEFT)

        self._grid_size_var = tk.StringVar(value=str(self.project.grid_size))
        self._snap_tol_var  = tk.StringVar(value=str(self.project.snap_tolerance))
        self._h_var         = tk.StringVar(value=str(self.project.export.wall_height))
        self._t_var         = tk.StringVar(value=str(self.project.export.wall_thickness))

        cfg_row("Rejilla (m):",  self._grid_size_var)
        cfg_row("Snap tol. (m):", self._snap_tol_var)
        cfg_row("Altura muro:",  self._h_var)
        cfg_row("Grosor muro:",  self._t_var)

        tk.Button(
            f, text="Actualizar config.",
            bg=C["btn"], fg=C["text"], font=_FONT_SM, relief=tk.FLAT,
            padx=4, pady=4,
            activebackground=C["btn_act"], activeforeground=C["text"],
            command=self._apply_config,
        ).pack(fill=tk.X, padx=8, pady=(4, 2))

        # Sección plano de fondo
        tk.Frame(f, bg=C["sep"], height=1).pack(fill=tk.X, padx=8, pady=(12, 2))
        tk.Label(f, text="PLANO DE FONDO", bg=C["bg_panel"], fg=C["text_dim"],
                 font=_FONT_SM).pack(pady=(2, 4), padx=10, anchor="w")

        self._bg_info_label = tk.Label(f, text="Sin plano cargado", bg=C["bg_panel"],
                                       fg=C["text_dim"], font=_FONT_SM, wraplength=200,
                                       justify=tk.LEFT)
        self._bg_info_label.pack(padx=10, anchor="w")

        op_row = tk.Frame(f, bg=C["bg_panel"])
        op_row.pack(fill=tk.X, padx=10, pady=(4, 0))
        tk.Label(op_row, text="Opacidad:", bg=C["bg_panel"], fg=C["text"],
                 font=_FONT_SM, width=9, anchor="w").pack(side=tk.LEFT)
        self._opacity_var = tk.DoubleVar(value=0.5)
        self._opacity_scale = tk.Scale(
            op_row, from_=0.05, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self._opacity_var,
            bg=C["bg_panel"], fg=C["text"], troughcolor=C["bg_input"],
            highlightthickness=0, bd=0, length=110,
            command=self._on_opacity_change,
        )
        self._opacity_scale.pack(side=tk.LEFT)

        tk.Button(
            f, text="Importar plano…",
            bg=C["btn"], fg=C["text"], font=_FONT_SM, relief=tk.FLAT,
            padx=4, pady=4,
            activebackground=C["accent"], activeforeground=C["bg"],
            command=self._import_background,
        ).pack(fill=tk.X, padx=8, pady=(4, 1))

        tk.Button(
            f, text="Detectar paredes…",
            bg="#1e2a1e", fg=C["success"], font=_FONT_SM, relief=tk.FLAT,
            padx=4, pady=4,
            activebackground=C["success"], activeforeground=C["bg"],
            command=self._open_detection_dialog,
        ).pack(fill=tk.X, padx=8, pady=1)

        tk.Button(
            f, text="Confirmar detectados",
            bg=C["btn"], fg=C["text"], font=_FONT_SM, relief=tk.FLAT,
            padx=4, pady=4,
            activebackground=C["btn_act"], activeforeground=C["text"],
            command=self._confirm_detected,
        ).pack(fill=tk.X, padx=8, pady=(1, 2))

        self._update_properties()

    def _build_statusbar(self) -> None:
        bar = tk.Frame(self.root, bg=C["bg_panel"], height=24)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._tool_label = tk.Label(
            bar, text="Herramienta: Dibujar",
            bg=C["bg_panel"], fg=C["accent"], font=_FONT_SM, padx=8
        )
        self._tool_label.pack(side=tk.LEFT)

        tk.Frame(bar, bg=C["sep"], width=1).pack(side=tk.LEFT, fill=tk.Y, pady=2)

        tk.Label(bar, textvariable=self.status_var,
                 bg=C["bg_panel"], fg=C["text"], font=_FONT_SM, padx=8
                 ).pack(side=tk.LEFT)

        self._walls_label = tk.Label(
            bar, text="0 muros",
            bg=C["bg_panel"], fg=C["text_dim"], font=_FONT_SM, padx=8
        )
        self._walls_label.pack(side=tk.RIGHT)

    # ── Callbacks de cambio ───────────────────────────────────────────────────

    def _on_project_change(self) -> None:
        self._update_title()
        self._update_properties()
        self._update_tool_buttons()
        self._update_bg_info()
        n = len(self.project.walls)
        s = len(self.canvas.selected_ids)
        nd = sum(1 for w in self.project.walls if w.state.value == "detected")
        self._walls_label.config(
            text=f"{n} muro{'s' if n != 1 else ''}"
                 + (f"  ({nd} pend.)" if nd else "")
        )
        self._info_walls.config(text=f"Muros: {n}  (det: {nd})")
        self._info_sel.config(text=f"Seleccionados: {s}")

    # ── Actualización de UI ───────────────────────────────────────────────────

    def _update_title(self) -> None:
        mod = " •" if self.project.modified else ""
        name = self.project.name
        self.root.title(f"{name}{mod} — {APP_TITLE}")

    def _update_tool_buttons(self) -> None:
        active = self.canvas.active_tool_name
        for tool, btn in self._tool_btns.items():
            btn.config(bg=C["accent"] if tool == active else C["btn"],
                       fg=C["bg"] if tool == active else C["text"])
        tool_names = {"select": "Seleccionar", "draw": "Dibujar", "delete": "Borrar"}
        self._tool_label.config(text=f"Herramienta: {tool_names.get(active, active)}")

    def _update_properties(self) -> None:
        sel = [self.project.get_wall(wid) for wid in self.canvas.selected_ids]
        sel = [w for w in sel if w is not None]

        if len(sel) == 1:
            w = sel[0]
            data = {
                "id":        w.id,
                "length":    f"{w.length():.3f} m",
                "angle":     f"{w.angle_deg():.2f}°",
                "thickness": str(round(w.thickness, 4)),
                "height":    str(round(w.height, 4)),
                "material":  w.material,
                "state":     w.state.value,
            }
            for key, val in data.items():
                widget = self._prop_labels[key]
                if isinstance(widget, tuple):
                    widget[0].set(val)
                else:
                    widget.config(text=val)
            self._apply_btn.config(state=tk.NORMAL)
        else:
            for key, widget in self._prop_labels.items():
                placeholder = "—" if len(sel) == 0 else f"({len(sel)} selec.)"
                if isinstance(widget, tuple):
                    widget[0].set(placeholder if key not in ("thickness", "height", "material") else "")
                else:
                    widget.config(text=placeholder)
            self._apply_btn.config(state=tk.DISABLED if len(sel) == 0 else tk.NORMAL)

        n = len(self.project.walls)
        self._info_walls.config(text=f"Muros: {n}")
        self._info_sel.config(text=f"Seleccionados: {len(sel)}")

    # ── Herramientas ─────────────────────────────────────────────────────────

    def _set_tool(self, name: str) -> None:
        self.canvas.set_tool(name)
        self._update_tool_buttons()

    # ── Acciones de proyecto ──────────────────────────────────────────────────

    def _new_project(self) -> None:
        if self.project.modified:
            if not messagebox.askyesno("Nuevo proyecto",
                                       "Hay cambios sin guardar. ¿Continuar?"):
                return
        name = simpledialog.askstring("Nuevo proyecto", "Nombre del proyecto:",
                                      initialvalue="Untitled", parent=self.root)
        if not name:
            return
        self.project = Project.new(name)
        self.canvas.project = self.project
        self.canvas.snap.grid_size   = self.project.grid_size
        self.canvas.snap.endpoint_tol = self.project.snap_tolerance
        self.canvas.selected_ids.clear()
        self.canvas._undo_stack.clear()
        self.canvas._redo_stack.clear()
        self.canvas.refresh()
        self._on_project_change()

    def _open_project(self) -> None:
        if self.project.modified:
            if not messagebox.askyesno("Abrir proyecto",
                                       "Hay cambios sin guardar. ¿Continuar?"):
                return
        path = filedialog.askopenfilename(
            title="Abrir proyecto WallForge",
            filetypes=[("WallForge Project", "*.wfp *.json"), ("Todos", "*.*")],
        )
        if not path:
            return
        try:
            self.project = Project.load(Path(path))
            self.canvas.project = self.project
            self.canvas.snap.grid_size   = self.project.grid_size
            self.canvas.snap.endpoint_tol = self.project.snap_tolerance
            self.canvas.selected_ids.clear()
            self.canvas._undo_stack.clear()
            self.canvas._redo_stack.clear()
            self.canvas.fit_to_content()
            self._on_project_change()
        except Exception as e:
            messagebox.showerror("Error al abrir", str(e))

    def _save_project(self) -> None:
        if self.project.file_path is None:
            self._save_as()
        else:
            try:
                self.project.save()
                self._update_title()
                self.status_var.set(f"Guardado: {self.project.file_path.name}")
            except Exception as e:
                messagebox.showerror("Error al guardar", str(e))

    def _save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Guardar proyecto como",
            defaultextension=".wfp",
            filetypes=[("WallForge Project", "*.wfp"), ("JSON", "*.json")],
            initialfile=f"{self.project.name}.wfp",
        )
        if not path:
            return
        try:
            self.project.name = Path(path).stem
            self.project.save(Path(path))
            self._update_title()
            self.status_var.set(f"Guardado: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Error al guardar", str(e))

    # ── Edición ───────────────────────────────────────────────────────────────

    def _undo(self) -> None:
        self.canvas.undo()
        self._on_project_change()

    def _redo(self) -> None:
        self.canvas.redo()
        self._on_project_change()

    def _delete_selected(self) -> None:
        self.canvas.delete_selected()
        self._on_project_change()

    def _select_all(self) -> None:
        self.canvas.selected_ids = {w.id for w in self.project.walls}
        self.canvas.refresh()
        self._on_project_change()

    def _deselect(self) -> None:
        self.canvas.deselect_all()
        self.canvas.refresh()
        self._update_properties()

    def _apply_properties(self) -> None:
        sel = [self.project.get_wall(wid) for wid in self.canvas.selected_ids]
        sel = [w for w in sel if w is not None]
        if not sel:
            return
        try:
            t_str = self._prop_labels["thickness"][0].get()
            h_str = self._prop_labels["height"][0].get()
            mat   = self._prop_labels["material"][0].get().strip()
            t = float(t_str) if t_str else None
            h = float(h_str) if h_str else None
        except ValueError:
            messagebox.showerror("Error", "Valores de grosor/altura inválidos.")
            return
        self.canvas._push_undo()
        for w in sel:
            if t and t > 0: w.thickness = t
            if h and h > 0: w.height    = h
            if mat:          w.material  = mat
            self.project.modified = True
        self.canvas.refresh()
        self._update_properties()

    def _apply_config(self) -> None:
        try:
            gs = float(self._grid_size_var.get())
            st = float(self._snap_tol_var.get())
            hh = float(self._h_var.get())
            tt = float(self._t_var.get())
        except ValueError:
            messagebox.showerror("Error", "Valores de configuración inválidos.")
            return
        self.project.grid_size                    = max(0.05, gs)
        self.project.snap_tolerance               = max(0.0, st)
        self.project.export.wall_height           = max(0.1, hh)
        self.project.export.wall_thickness        = max(0.01, tt)
        self.canvas.snap.grid_size               = self.project.grid_size
        self.canvas.snap.endpoint_tol            = self.project.snap_tolerance
        self.canvas.refresh()
        self.status_var.set("Configuración actualizada.")

    # ── Vista ─────────────────────────────────────────────────────────────────

    def _fit(self) -> None:
        self.canvas.fit_to_content()

    def _center(self) -> None:
        self.canvas._center_origin()

    def _toggle_grid(self) -> None:
        on = self._grid_var.get()
        self.canvas.snap.grid_enabled = on
        self.canvas.project.grid_size = self.project.grid_size if on else 0.0
        self.canvas.refresh()

    def _toggle_snap(self) -> None:
        on = self._snap_var.get()
        self.canvas.snap.grid_enabled     = on
        self.canvas.snap.endpoint_enabled = on

    # ── Exportación ──────────────────────────────────────────────────────────

    def _export_basic(self) -> None:
        if not self.project.walls:
            messagebox.showwarning("Exportar", "No hay muros para exportar.")
            return
        path = filedialog.asksaveasfilename(
            title="Exportar MuJoCo XML básico",
            defaultextension=".xml",
            filetypes=[("MuJoCo XML", "*.xml"), ("Todos", "*.*")],
            initialfile=f"{self.project.name}_scene.xml",
        )
        if not path:
            return
        try:
            summary = export_basic(self.project.walls, self.project.export, Path(path))
            self._show_export_summary(summary)
        except Exception as e:
            messagebox.showerror("Error de exportación", str(e))

    def _export_repo(self) -> None:
        if not self.project.walls:
            messagebox.showwarning("Exportar", "No hay muros para exportar.")
            return
        path = filedialog.asksaveasfilename(
            title="Exportar para repo g1man",
            defaultextension=".xml",
            filetypes=[("MuJoCo XML", "*.xml"), ("Todos", "*.*")],
            initialfile=f"{self.project.name}_g1man.xml",
        )
        if not path:
            return
        try:
            summary = export_repo_scene(self.project.walls, self.project.export, Path(path))
            self._show_export_summary(summary)
        except Exception as e:
            messagebox.showerror("Error de exportación", str(e))

    def _show_export_summary(self, summary: dict) -> None:
        lines = [
            f"  Modo:         {summary.get('mode', '—')}",
            f"  Muros:        {summary.get('walls', 0)}",
            f"  Archivo:      {Path(summary.get('file', '')).name}",
        ]
        if "robot" in summary:
            lines.append(f"  Robot:        {summary['robot']}")
        if "extent" in summary:
            lines.append(f"  Extent:       {summary['extent']:.0f} m")
        if "instructions" in summary:
            lines.append("")
            lines.append("  " + summary["instructions"].replace("\n", "\n  "))
        messagebox.showinfo("Exportación completada", "\n".join(lines))

    # ── Atajos y ayuda ────────────────────────────────────────────────────────

    def _bind_shortcuts(self) -> None:
        r = self.root
        r.bind("<Control-n>", lambda e: self._new_project())
        r.bind("<Control-o>", lambda e: self._open_project())
        r.bind("<Control-s>", lambda e: self._save_project())
        r.bind("<Control-S>", lambda e: self._save_as())
        r.bind("<Control-q>", lambda e: self._quit())
        r.bind("<Control-z>", lambda e: self._undo())
        r.bind("<Control-y>", lambda e: self._redo())
        r.bind("<Control-Z>", lambda e: self._redo())  # Ctrl+Shift+Z
        r.bind("<Delete>",    lambda e: self._delete_selected())
        r.bind("<BackSpace>", lambda e: self._delete_selected())
        r.bind("<Control-a>", lambda e: self._select_all())
        r.bind("<s>",         lambda e: self._set_tool("select"))
        r.bind("<d>",         lambda e: self._set_tool("draw"))
        r.bind("<x>",         lambda e: self._set_tool("delete"))
        r.bind("<f>",         lambda e: self._fit())
        r.bind("<Home>",      lambda e: self._center())
        r.bind("<Control-g>", lambda e: self._toggle_snap_kb())
        r.bind("<Control-i>", lambda e: self._import_background())
        r.bind("<Control-d>", lambda e: self._open_detection_dialog())

    def _toggle_snap_kb(self) -> None:
        self._snap_var.set(not self._snap_var.get())
        self._toggle_snap()

    # ── Plano de fondo ────────────────────────────────────────────────────────

    def _import_background(self) -> None:
        path = filedialog.askopenfilename(
            title="Importar plano arquitectónico",
            filetypes=[
                ("Imágenes y PDF", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.pdf"),
                ("PNG",  "*.png"), ("JPEG", "*.jpg *.jpeg"),
                ("PDF",  "*.pdf"), ("Todos", "*.*"),
            ],
        )
        if not path:
            return
        try:
            from ..model.project import BackgroundLayer
            bg = BackgroundLayer(image_path=path, ppm=100.0,
                                 origin_world=(0.0, 0.0), opacity=0.5)
            self.status_var.set("Cargando imagen…")
            self.root.update_idletasks()
            bg.load_image()
            self.project.background = bg
            self.project.modified = True
            self._calibrate_background(auto_open=True)
            self._update_bg_info()
            self.canvas.refresh()
            self.status_var.set(
                f"Plano cargado: {Path(path).name}  "
                f"({bg.img_size[0]}×{bg.img_size[1]} px, {bg.ppm:.0f} px/m)"
            )
        except Exception as e:
            messagebox.showerror("Error al importar", str(e))

    def _calibrate_background(self, auto_open: bool = False) -> None:
        bg = self.project.background
        if bg is None:
            messagebox.showwarning("Calibración", "Primero importa un plano de fondo.")
            return
        if not bg.loaded:
            messagebox.showwarning("Calibración", "La imagen no está cargada aún.")
            return

        win = tk.Toplevel(self.root)
        win.title("Calibración de escala — WallForge")
        win.configure(bg=C["bg"])
        win.grab_set()
        win.resizable(True, True)

        tk.Label(win,
                 text="Dibuja líneas sobre el plano y escribe la distancia real de cada una.",
                 bg=C["bg"], fg=C["text"], font=_FONT_SM, pady=6).pack()

        # ── body: left=imagen, right=panel ───────────────────────────────────
        body = tk.Frame(win, bg=C["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=12)

        left = tk.Frame(body, bg=C["bg"])
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = tk.Frame(body, bg=C["bg"], width=250)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=(14, 0))
        right.pack_propagate(False)

        # ── Imagen en canvas ──────────────────────────────────────────────────
        MAX_W, MAX_H = 620, 460
        img_w, img_h = bg.img_size
        scale = min(MAX_W / max(img_w, 1), MAX_H / max(img_h, 1))
        disp_w = max(1, int(img_w * scale))
        disp_h = max(1, int(img_h * scale))

        try:
            from PIL import Image as _PilImg, ImageTk
            _rsmp = getattr(_PilImg, "LANCZOS", None) or _PilImg.Resampling.LANCZOS
            disp_img = bg._pil_rgb.resize((disp_w, disp_h), _rsmp)
            photo = ImageTk.PhotoImage(disp_img)
        except Exception as exc:
            messagebox.showerror("Error", f"No se pudo mostrar la imagen:\n{exc}", parent=win)
            win.destroy()
            return

        img_cv = tk.Canvas(left, width=disp_w, height=disp_h,
                           bg="#0d0d18", cursor="crosshair",
                           highlightthickness=1, highlightbackground=C["sep"])
        img_cv.pack(pady=4)
        img_cv.create_image(0, 0, anchor="nw", image=photo)
        img_cv._photo_ref = photo  # prevent GC

        hint_lbl = tk.Label(left,
                            text="1er clic: punto inicial  |  2º clic: punto final",
                            bg=C["bg"], fg=C["text_dim"], font=_FONT_SM)
        hint_lbl.pack(anchor="w")

        # ── Panel derecho ─────────────────────────────────────────────────────
        tk.Label(right, text="MEDICIONES", bg=C["bg"], fg=C["text_dim"],
                 font=_FONT_SM).pack(anchor="w", pady=(4, 2))
        tk.Frame(right, bg=C["sep"], height=1).pack(fill=tk.X, pady=2)

        meas_frame = tk.Frame(right, bg=C["bg"])
        meas_frame.pack(fill=tk.X)

        avg_lbl = tk.Label(right, text="Escala media: —",
                           bg=C["bg"], fg=C["accent"], font=_FONT_BOLD)
        avg_lbl.pack(anchor="w", pady=(8, 4))

        tk.Frame(right, bg=C["sep"], height=1).pack(fill=tk.X, pady=6)
        tk.Label(right, text="O introduce la escala directamente (px/m):",
                 bg=C["bg"], fg=C["text"], font=_FONT_SM, wraplength=220,
                 justify=tk.LEFT).pack(anchor="w")
        ppm_var = tk.StringVar(value=f"{bg.ppm:.2f}")
        ppm_entry = tk.Entry(right, textvariable=ppm_var, bg=C["bg_input"],
                             fg=C["text"], insertbackground=C["text"],
                             relief=tk.FLAT, font=_FONT_MONO, width=12)
        ppm_entry.pack(anchor="w", pady=2)
        tk.Label(right, text="Ej: 50=2cm/px  100=1cm/px  200=5mm/px",
                 bg=C["bg"], fg=C["text_dim"], font=_FONT_SM).pack(anchor="w")

        # ── Estado interno ────────────────────────────────────────────────────
        measures: list = []     # [(px_dist, real_m, ppm)]
        drawn:    list = []     # [list of canvas item ids per measure]
        pending = {"p1": None, "dot1": None, "ghost": None}

        def _refresh_panel():
            for w in meas_frame.winfo_children():
                w.destroy()
            for i, (px_d, real_m, ppm_v) in enumerate(measures):
                row = tk.Frame(meas_frame, bg=C["bg"])
                row.pack(fill=tk.X, pady=1)
                tk.Label(row,
                         text=f"#{i+1}  {px_d:.0f}px / {real_m}m → {ppm_v:.1f}px/m",
                         bg=C["bg"], fg=C["text"], font=_FONT_SM).pack(side=tk.LEFT)
                def _rm(idx=i):
                    for item in drawn[idx]:
                        img_cv.delete(item)
                    measures.pop(idx)
                    drawn.pop(idx)
                    _refresh_panel()
                tk.Button(row, text="✕", bg=C["bg"], fg=C["danger"],
                          font=_FONT_SM, relief=tk.FLAT, padx=2, pady=0,
                          command=_rm).pack(side=tk.RIGHT)
            if measures:
                avg = sum(m[2] for m in measures) / len(measures)
                avg_lbl.config(text=f"Escala media: {avg:.1f} px/m")
                ppm_var.set(f"{avg:.2f}")
            else:
                avg_lbl.config(text="Escala media: —")

        def _ask_distance(px_dist, items, x1, y1, x2, y2):
            """Inline widget in right panel to capture real-world distance."""
            ask_f = tk.Frame(right, bg=C["bg_input"], relief=tk.FLAT, bd=1)
            ask_f.pack(fill=tk.X, pady=4, after=meas_frame)

            tk.Label(ask_f, text=f"Línea medida: {px_dist:.0f} px",
                     bg=C["bg_input"], fg=C["text"], font=_FONT_SM).pack(anchor="w", padx=6)
            tk.Label(ask_f, text="Distancia real (metros):",
                     bg=C["bg_input"], fg=C["text"], font=_FONT_SM).pack(anchor="w", padx=6)
            dist_var = tk.StringVar()
            dist_ent = tk.Entry(ask_f, textvariable=dist_var, bg=C["bg_input"],
                                fg=C["text"], insertbackground=C["text"],
                                relief=tk.FLAT, font=_FONT_MONO, width=10)
            dist_ent.pack(anchor="w", padx=6, pady=2)
            dist_ent.focus_set()
            err_lbl = tk.Label(ask_f, text="", bg=C["bg_input"],
                               fg=C["danger"], font=_FONT_SM)
            err_lbl.pack(anchor="w", padx=6)

            def _confirm():
                try:
                    real_m = float(dist_var.get().replace(",", "."))
                    if real_m <= 0:
                        raise ValueError
                    ppm_v = px_dist / real_m
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    lbl_id = img_cv.create_text(
                        mid_x, mid_y - 14,
                        text=f"{real_m}m ({ppm_v:.0f}px/m)",
                        fill=C["success"], font=_FONT_SM,
                    )
                    measures.append((px_dist, real_m, ppm_v))
                    drawn.append(items + [lbl_id])
                    ask_f.destroy()
                    _refresh_panel()
                    hint_lbl.config(
                        text="1er clic: punto inicial  |  2º clic: punto final")
                except ValueError:
                    err_lbl.config(text="Valor inválido — usa punto decimal")

            def _cancel():
                for item in items:
                    img_cv.delete(item)
                ask_f.destroy()
                hint_lbl.config(
                    text="1er clic: punto inicial  |  2º clic: punto final")

            btn_r = tk.Frame(ask_f, bg=C["bg_input"])
            btn_r.pack(fill=tk.X, padx=6, pady=(2, 4))
            tk.Button(btn_r, text="OK", bg="#1e3a1e", fg=C["success"],
                      font=_FONT_SM, relief=tk.FLAT, padx=8, pady=2,
                      command=_confirm).pack(side=tk.LEFT, padx=(0, 4))
            tk.Button(btn_r, text="✕ Cancelar", bg=C["btn"], fg=C["text"],
                      font=_FONT_SM, relief=tk.FLAT, padx=6, pady=2,
                      command=_cancel).pack(side=tk.LEFT)
            dist_ent.bind("<Return>", lambda e: _confirm())
            dist_ent.bind("<Escape>", lambda e: _cancel())

        def _on_motion(event):
            if pending["p1"] is None:
                return
            if pending["ghost"]:
                img_cv.delete(pending["ghost"])
            x1, y1 = pending["p1"]
            pending["ghost"] = img_cv.create_line(
                x1, y1, event.x, event.y,
                fill=C["accent"], width=1, dash=(5, 3),
            )

        def _on_click(event):
            x, y = event.x, event.y
            if pending["p1"] is None:
                dot = img_cv.create_oval(x-5, y-5, x+5, y+5,
                                         fill=C["accent"], outline="white", width=1)
                pending["p1"]   = (x, y)
                pending["dot1"] = dot
                hint_lbl.config(text="2º clic para fijar el punto final…")
            else:
                x1, y1 = pending["p1"]
                if pending["ghost"]:
                    img_cv.delete(pending["ghost"])
                    pending["ghost"] = None
                line = img_cv.create_line(x1, y1, x, y, fill=C["accent"], width=2)
                dot2 = img_cv.create_oval(x-5, y-5, x+5, y+5,
                                          fill=C["accent"], outline="white", width=1)
                px_dist = math.hypot(x - x1, y - y1) / scale
                items   = [pending["dot1"], line, dot2]
                pending["p1"]   = None
                pending["dot1"] = None
                _ask_distance(px_dist, items, x1, y1, x, y)

        img_cv.bind("<Button-1>", _on_click)
        img_cv.bind("<Motion>",   _on_motion)

        # ── Botones finales ───────────────────────────────────────────────────
        def apply():
            try:
                bg.ppm = max(1.0, float(ppm_var.get()))
                self.project.modified = True
                self._update_bg_info()
                self.canvas.refresh()
                win.destroy()
            except ValueError:
                messagebox.showerror("Error", "Valor de px/m inválido.", parent=win)

        btn_row = tk.Frame(win, bg=C["bg"])
        btn_row.pack(pady=(8, 12))
        tk.Button(btn_row, text="Aplicar", bg="#1e3a1e", fg=C["success"],
                  font=_FONT_BOLD, relief=tk.FLAT, padx=16, pady=6,
                  command=apply).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_row, text="Cancelar", bg=C["btn"], fg=C["text"],
                  font=_FONT_SM, relief=tk.FLAT, padx=12, pady=6,
                  command=win.destroy).pack(side=tk.LEFT)

        win.bind("<Return>", lambda e: apply())

    def _toggle_bg_visible(self) -> None:
        if self.project.background:
            self.project.background.visible = self._bg_visible_var.get()
            self.canvas.refresh()

    def _toggle_bg_locked(self) -> None:
        if self.project.background:
            self.project.background.locked = self._bg_locked_var.get()

    def _on_opacity_change(self, _=None) -> None:
        if self.project.background:
            self.project.background.opacity = self._opacity_var.get()
            self.canvas.refresh()

    def _remove_background(self) -> None:
        if self.project.background is None:
            return
        if messagebox.askyesno("Eliminar plano", "¿Eliminar el plano de fondo?"):
            self.project.background = None
            self.project.modified   = True
            self._update_bg_info()
            self.canvas.refresh()

    def _update_bg_info(self) -> None:
        bg = self.project.background
        if bg is None:
            self._bg_info_label.config(text="Sin plano cargado", fg=C["text_dim"])
        else:
            w, h = bg.img_size
            name = Path(bg.image_path).name
            self._bg_info_label.config(
                text=f"{name}\n{w}×{h} px  |  {bg.ppm:.0f} px/m",
                fg=C["text"]
            )
            self._opacity_var.set(bg.opacity)

    def _confirm_detected(self) -> None:
        """Convierte todos los muros DETECTED a CONFIRMED."""
        count = 0
        for w in self.project.walls:
            if w.state == WallState.DETECTED:
                w.state = WallState.CONFIRMED
                count += 1
        if count:
            self.project.modified = True
            self.canvas.refresh()
            self.status_var.set(f"{count} muros confirmados.")
        else:
            self.status_var.set("No hay muros detectados pendientes.")

    # ── Diálogo de detección ──────────────────────────────────────────────────

    def _open_detection_dialog(self) -> None:
        bg = self.project.background
        if bg is None or not bg.loaded:
            messagebox.showwarning("Detectar",
                                   "Primero importa un plano de fondo (Plano → Importar imagen).")
            return

        win = tk.Toplevel(self.root)
        win.title("Detectar paredes del plano")
        win.configure(bg=C["bg"])
        win.resizable(False, False)
        win.grab_set()

        def mk_label(parent, text, fg=None, font=None, **kw):
            return tk.Label(parent, text=text, bg=C["bg"],
                            fg=fg or C["text"], font=font or _FONT_SM, **kw)

        def param_row(parent, label, var, from_, to_, res):
            row = tk.Frame(parent, bg=C["bg"]); row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, bg=C["bg"], fg=C["text"],
                     font=_FONT_SM, width=22, anchor="w").pack(side=tk.LEFT)
            tk.Scale(row, from_=from_, to=to_, resolution=res,
                     orient=tk.HORIZONTAL, variable=var,
                     bg=C["bg"], fg=C["text"], troughcolor=C["bg_input"],
                     highlightthickness=0, bd=0, length=140).pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, bg=C["bg"], fg=C["accent"],
                     font=_FONT_MONO, width=6).pack(side=tk.LEFT)

        mk_label(win, "Parámetros de detección de líneas", pady=8,
                 font=_FONT_BOLD).pack()

        frame = tk.Frame(win, bg=C["bg"], padx=16); frame.pack(fill=tk.X)

        dark_var  = tk.IntVar(value=80)
        blur_var  = tk.IntVar(value=3)
        close_var = tk.IntVar(value=5)
        minl_var  = tk.IntVar(value=30)
        gap_var   = tk.IntVar(value=15)
        merge_var = tk.DoubleVar(value=10.0)
        invert_var = tk.BooleanVar(value=False)
        text_var  = tk.BooleanVar(value=True)

        param_row(frame, "Umbral oscuridad:",   dark_var,  20,  200, 5)
        param_row(frame, "Blur (px):",          blur_var,  1,   11,  2)
        param_row(frame, "Cierre morfoló. (px):", close_var, 0, 15,  1)
        param_row(frame, "Longitud mínima (px):", minl_var, 10, 200, 5)
        param_row(frame, "Hueco máximo (px):",  gap_var,   0,   50,  5)
        param_row(frame, "Fusión (px):",        merge_var, 2.0, 40.0, 1.0)

        checks = tk.Frame(win, bg=C["bg"]); checks.pack(fill=tk.X, padx=16, pady=4)
        tk.Checkbutton(checks, text="Plano invertido (fondo negro)",
                       variable=invert_var, bg=C["bg"], fg=C["text"],
                       selectcolor=C["bg_input"], activebackground=C["bg"],
                       font=_FONT_SM).pack(anchor="w")
        tk.Checkbutton(checks, text="Filtrar ruido de texto",
                       variable=text_var, bg=C["bg"], fg=C["text"],
                       selectcolor=C["bg_input"], activebackground=C["bg"],
                       font=_FONT_SM).pack(anchor="w")

        status_lbl = tk.Label(win, text="Listo.", bg=C["bg"], fg=C["text_dim"],
                              font=_FONT_SM, pady=4)
        status_lbl.pack()

        _last_walls = []

        def run_detection():
            from ..image.detector import DetectionParams, detect_walls
            p = DetectionParams(
                blur        = blur_var.get(),
                dark_thresh = dark_var.get(),
                closing     = close_var.get(),
                invert      = invert_var.get(),
                min_line    = minl_var.get(),
                max_gap     = gap_var.get(),
                merge_dist  = merge_var.get(),
                filter_text = text_var.get(),
                min_length_m = 0.1,
            )
            status_lbl.config(text="Detectando…", fg=C["accent"])
            win.update_idletasks()
            try:
                walls, n_raw = detect_walls(
                    bg._np_bgr, p,
                    ppm            = bg.ppm,
                    origin_world   = bg.origin_world,
                    wall_thickness = self.project.export.wall_thickness,
                    wall_height    = self.project.export.wall_height,
                )
                _last_walls.clear()
                _last_walls.extend(walls)
                status_lbl.config(
                    text=f"Detectados: {len(walls)} muros  (agrupados de {n_raw} crudos)",
                    fg=C["success"]
                )
                add_btn.config(state=tk.NORMAL)
            except Exception as e:
                status_lbl.config(text=f"Error: {e}", fg=C["danger"])

        def add_to_map():
            if not _last_walls:
                return
            self.canvas._push_undo()
            for w in _last_walls:
                self.project.add_wall(w)
            self.canvas.notify_change()
            self.canvas.refresh()
            self._on_project_change()
            win.destroy()
            self.status_var.set(
                f"{len(_last_walls)} muros detectados añadidos  "
                f"(amarillo = pendiente de confirmación)."
            )

        btn_row = tk.Frame(win, bg=C["bg"]); btn_row.pack(pady=(4, 12))
        tk.Button(btn_row, text="Detectar", bg="#1e2a3e", fg=C["accent"],
                  font=_FONT_BOLD, relief=tk.FLAT, padx=14, pady=6,
                  command=run_detection).pack(side=tk.LEFT, padx=6)
        add_btn = tk.Button(btn_row, text="Añadir al mapa", bg="#1e3a1e", fg=C["success"],
                            font=_FONT_BOLD, relief=tk.FLAT, padx=14, pady=6,
                            state=tk.DISABLED, command=add_to_map)
        add_btn.pack(side=tk.LEFT, padx=6)
        tk.Button(btn_row, text="Cancelar", bg=C["btn"], fg=C["text"],
                  font=_FONT_SM, relief=tk.FLAT, padx=12, pady=6,
                  command=win.destroy).pack(side=tk.LEFT)

    def _show_shortcuts(self) -> None:
        text = (
            "Atajos de teclado — WallForge Studio\n"
            "─────────────────────────────────────\n"
            "S           Herramienta Seleccionar\n"
            "D           Herramienta Dibujar\n"
            "X           Herramienta Borrar\n"
            "Esc         Cancelar acción / Deseleccionar\n"
            "Del/Back    Eliminar selección\n"
            "Ctrl+Z      Deshacer\n"
            "Ctrl+Y      Rehacer\n"
            "Ctrl+A      Seleccionar todo\n"
            "Ctrl+N      Nuevo proyecto\n"
            "Ctrl+O      Abrir proyecto\n"
            "Ctrl+S      Guardar\n"
            "Ctrl+G      Toggle snap\n"
            "F           Ajustar vista al contenido\n"
            "Home        Centrar origen\n"
            "Rueda       Zoom\n"
            "Botón medio / derecho + arrastrar   Pan\n"
            "Shift (al dibujar)   Snap de ángulo (15°)\n"
        )
        win = tk.Toplevel(self.root)
        win.title("Atajos de teclado")
        win.configure(bg=C["bg"])
        win.resizable(False, False)
        tk.Label(win, text=text, bg=C["bg"], fg=C["text"],
                 font=_FONT_MONO, justify=tk.LEFT, padx=20, pady=16).pack()
        tk.Button(win, text="Cerrar", bg=C["btn"], fg=C["text"],
                  font=_FONT, relief=tk.FLAT, padx=12, pady=4,
                  command=win.destroy).pack(pady=(0, 12))

    def _show_about(self) -> None:
        messagebox.showinfo(
            f"Acerca de {APP_TITLE}",
            f"{APP_TITLE}  v{VERSION}\n\n"
            "Editor de mundos para simulación robótica.\n"
            "Compatible con MuJoCo/MJCF y el repositorio g1man.\n\n"
            "Fase 1 — Base del editor."
        )

    def _quit(self) -> None:
        if self.project.modified:
            if not messagebox.askyesno("Salir", "Hay cambios sin guardar. ¿Salir de todos modos?"):
                return
        self.root.destroy()

    # ── Entrada ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.mainloop()
