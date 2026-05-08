"""
Pipeline de detección de paredes a partir de una imagen de plano.
Portado y limpiado de image_to_mujoco.py.

Flujo:
  1. build_wall_mask()       → máscara binaria de píxeles oscuros (paredes)
  2. remove_text_noise()     → elimina componentes pequeños (texto, ruido)
  3. detect_hough()          → segmentos Hough
  4. detect_components()     → rectángulos de componentes conectadas
  5. merge_colinear()        → fusión de segmentos colineales
  6. px_walls_to_world()     → conversión píxeles → metros (Wall del editor)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..model.wall import Wall, WallState


# ── Modelo interno de pared en píxeles ────────────────────────────────────────

@dataclass
class _PixWall:
    cx:        float   # centro X en la imagen (px)
    cy:        float   # centro Y en la imagen (px)
    length:    float   # longitud (px)
    thickness: float   # grosor estimado (px)
    angle_deg: float   # ángulo en coordenadas de imagen (y hacia abajo)
    source:    str = "unknown"

    def normalized_angle(self) -> float:
        return self.angle_deg % 180.0

    def direction(self) -> Tuple[float, float]:
        a = math.radians(self.angle_deg)
        return math.cos(a), math.sin(a)

    def endpoints(self) -> Tuple[Tuple, Tuple]:
        ux, uy = self.direction()
        hl = self.length / 2.0
        return (self.cx - ux * hl, self.cy - uy * hl), (self.cx + ux * hl, self.cy + uy * hl)


# ══════════════════════════════════════════════════════════════════════════════
# Paso 1 – Máscara de paredes
# ══════════════════════════════════════════════════════════════════════════════

def build_wall_mask(
    img_bgr: np.ndarray,
    blur:        int  = 3,
    dark_thresh: int  = 80,
    closing:     int  = 5,
    invert:      bool = False,
) -> np.ndarray:
    """Devuelve máscara binaria (255 = pared) robusta ante ruido y JPEG."""
    work = img_bgr if not invert else cv2.bitwise_not(img_bgr)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    if blur > 1:
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    _, otsu  = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    v        = cv2.split(cv2.cvtColor(work, cv2.COLOR_BGR2HSV))[2]
    dark_v   = (v  <= dark_thresh).astype(np.uint8) * 255
    lab_l    = cv2.split(cv2.cvtColor(work, cv2.COLOR_BGR2LAB))[0]
    dark_l   = (lab_l <= dark_thresh).astype(np.uint8) * 255

    mask = cv2.bitwise_or(cv2.bitwise_or(otsu, dark_v), dark_l)

    k3   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3)

    if closing > 0:
        kc   = closing if closing % 2 == 1 else closing + 1
        ks   = cv2.getStructuringElement(cv2.MORPH_RECT, (kc, kc))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ks)

    return mask


# ══════════════════════════════════════════════════════════════════════════════
# Paso 2 – Eliminar ruido de texto
# ══════════════════════════════════════════════════════════════════════════════

def remove_text_noise(
    mask:        np.ndarray,
    max_area:    int   = 1200,
    max_aspect:  float = 5.0,
    min_wall_dim: int  = 40,
) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        mx = max(w, h)
        aspect = mx / max(min(w, h), 1)
        if mx >= min_wall_dim or area >= max_area or aspect >= max_aspect:
            out[labels == i] = 255
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Paso 3 – Detección Hough
# ══════════════════════════════════════════════════════════════════════════════

def detect_hough(
    mask:      np.ndarray,
    min_len:   int = 30,
    max_gap:   int = 15,
    threshold: int = 25,
) -> List[_PixWall]:
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                             minLineLength=min_len, maxLineGap=max_gap)
    walls: List[_PixWall] = []
    if lines is None:
        return walls
    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, ln)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < min_len:
            continue
        cx    = (x1 + x2) / 2
        cy    = (y1 + y2) / 2
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        walls.append(_PixWall(cx, cy, length, 4.0, angle, "hough"))
    return walls


# ══════════════════════════════════════════════════════════════════════════════
# Paso 4 – Detección por componentes conectadas
# ══════════════════════════════════════════════════════════════════════════════

def detect_components(
    mask:      np.ndarray,
    min_area:  int = 80,
    min_len:   int = 25,
    max_thick: int = 120,
) -> List[_PixWall]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    walls: List[_PixWall] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        m   = np.uint8(labels == i) * 255
        pts = cv2.findNonZero(m)
        if pts is None or len(pts) < 5:
            continue
        rect = cv2.minAreaRect(pts)
        (cx, cy), (rw, rh), angle = rect
        length = max(rw, rh)
        thick  = max(1.0, min(rw, rh))
        if length < min_len or thick > max_thick:
            continue
        if rw < rh:
            angle += 90.0
        walls.append(_PixWall(cx, cy, length, thick, angle, "cc"))
    return walls


# ══════════════════════════════════════════════════════════════════════════════
# Paso 5 – Fusión de segmentos colineales
# ══════════════════════════════════════════════════════════════════════════════

def _angle_diff(a: float, b: float) -> float:
    d = abs((a % 180) - (b % 180))
    return min(d, 180 - d)


def _cluster_to_wall(cluster: List[_PixWall]) -> _PixWall:
    if len(cluster) == 1:
        c = cluster[0]
        return _PixWall(c.cx, c.cy, c.length, c.thickness, c.angle_deg, c.source)
    angles  = np.array([w.normalized_angle() for w in cluster], dtype=np.float64)
    rad     = np.deg2rad(angles)
    mean_a  = math.degrees(math.atan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))) % 180.0
    ux, uy  = math.cos(math.radians(mean_a)), math.sin(math.radians(mean_a))
    nx, ny  = -uy, ux
    points: List[Tuple] = []
    thicknesses = []
    for w in cluster:
        points.extend(w.endpoints())
        thicknesses.append(w.thickness)
    projs  = [x * ux + y * uy for x, y in points]
    norms  = [x * nx + y * ny for x, y in points]
    pmin, pmax = min(projs), max(projs)
    dmean  = float(np.mean(norms))
    cx = (pmin + pmax) / 2 * ux + dmean * nx
    cy = (pmin + pmax) / 2 * uy + dmean * ny
    return _PixWall(cx, cy, pmax - pmin, float(np.median(thicknesses)), mean_a, "merged")


def merge_colinear(
    walls:     List[_PixWall],
    dist_tol:  float = 10.0,
    angle_tol: float = 5.0,
    gap_tol:   float = 25.0,
) -> List[_PixWall]:
    if not walls:
        return []
    used   = [False] * len(walls)
    merged: List[_PixWall] = []
    for i, wi in enumerate(walls):
        if used[i]:
            continue
        used[i]  = True
        cluster  = [wi]
        changed  = True
        while changed:
            changed = False
            ref     = _cluster_to_wall(cluster)
            ang_r   = math.radians(ref.normalized_angle())
            ux, uy  = math.cos(ang_r), math.sin(ang_r)
            nx, ny  = -uy, ux
            d_ref   = nx * ref.cx + ny * ref.cy
            projs_r = []
            for w in cluster:
                (x1, y1), (x2, y2) = w.endpoints()
                projs_r += [x1*ux + y1*uy, x2*ux + y2*uy]
            pmin, pmax = min(projs_r), max(projs_r)
            for j, wj in enumerate(walls):
                if used[j]:
                    continue
                if _angle_diff(ref.angle_deg, wj.angle_deg) > angle_tol:
                    continue
                d_j = nx * wj.cx + ny * wj.cy
                if abs(d_ref - d_j) > dist_tol:
                    continue
                eps = [p[0]*ux + p[1]*uy for p in wj.endpoints()]
                if min(eps) <= pmax + gap_tol and max(eps) >= pmin - gap_tol:
                    cluster.append(wj)
                    used[j] = True
                    changed = True
        merged.append(_cluster_to_wall(cluster))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Conversión píxeles → muros del editor (metros, coordenadas mundo)
# ══════════════════════════════════════════════════════════════════════════════

def px_wall_to_world(
    pw:           _PixWall,
    img_w:        int,
    img_h:        int,
    ppm:          float,
    origin_world: Tuple[float, float],
    wall_thickness: float = 0.15,
    wall_height:    float = 2.5,
) -> Wall:
    """Convierte una _PixWall (píxeles) a Wall del editor (metros)."""
    # Centro en mundo (el eje Y se invierte: imagen y↓, mundo y↑)
    mx_w = origin_world[0] + (pw.cx - img_w / 2.0) / ppm
    my_w = origin_world[1] - (pw.cy - img_h / 2.0) / ppm

    # Ángulo en mundo: invertir Y → negar el ángulo
    angle_rad = -math.radians(pw.angle_deg)
    hl        = pw.length / (2.0 * ppm)      # mitad de longitud en metros

    p1 = (mx_w - math.cos(angle_rad) * hl,
          my_w - math.sin(angle_rad) * hl)
    p2 = (mx_w + math.cos(angle_rad) * hl,
          my_w + math.sin(angle_rad) * hl)

    return Wall(p1=p1, p2=p2,
                thickness=wall_thickness,
                height=wall_height,
                state=WallState.DETECTED)


# ══════════════════════════════════════════════════════════════════════════════
# Función principal de detección
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectionParams:
    blur:        int   = 3
    dark_thresh: int   = 80
    closing:     int   = 5
    invert:      bool  = False
    min_line:    int   = 30    # px mínimos Hough
    max_gap:     int   = 15    # px hueco máximo Hough
    hough_thr:   int   = 25
    merge_dist:  float = 10.0
    min_length_m: float = 0.1  # metros mínimos del muro resultante
    filter_text: bool  = True


def detect_walls(
    img_bgr:      np.ndarray,
    params:       DetectionParams,
    ppm:          float,
    origin_world: Tuple[float, float],
    wall_thickness: float = 0.15,
    wall_height:    float = 2.5,
) -> Tuple[List[Wall], int]:
    """
    Pipeline completo: imagen → List[Wall] en coordenadas mundo.
    Devuelve (muros, n_px_walls_antes_de_filtro).
    """
    H, W = img_bgr.shape[:2]

    # 1. Máscara
    mask = build_wall_mask(img_bgr, params.blur, params.dark_thresh,
                           params.closing, params.invert)
    # 2. Filtro de texto
    if params.filter_text:
        mask = remove_text_noise(mask)

    # 3. Detección
    w_hough = detect_hough(mask, params.min_line, params.max_gap, params.hough_thr)
    w_cc    = detect_components(mask)

    # 4. Fusión
    px_walls = merge_colinear(w_hough + w_cc, params.merge_dist)
    n_raw    = len(px_walls)

    # 5. Filtro de longitud mínima
    min_len_px = params.min_length_m * ppm
    px_walls = [w for w in px_walls if w.length >= max(15.0, min_len_px * 0.5)]

    # 6. Conversión a mundo
    world_walls = [
        px_wall_to_world(pw, W, H, ppm, origin_world, wall_thickness, wall_height)
        for pw in px_walls
        if pw.length >= 15.0
    ]

    return world_walls, n_raw
