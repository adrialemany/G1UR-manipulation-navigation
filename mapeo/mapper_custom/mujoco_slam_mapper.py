#!/usr/bin/env python3
"""
mujoco_slam_mapper.py
=====================
Mapeador de rejilla de ocupacion (Occupancy Grid) para ROS 2 Humble.
MAPA DINAMICO SIN LIMITE: crece en cualquier direccion segun el robot explora.

Suscribe:
    /scan   (sensor_msgs/LaserScan)
    /odom   (nav_msgs/Odometry)

Publica:
    /map    (nav_msgs/OccupancyGrid)

Teclas en terminal:
    m  — Guardar mapa en maps/
    r  — Resetear
    q  — Salir

Uso:
    cd mujoco/simulacion && python3 mujoco_slam_mapper.py
"""

import math
import os
import sys
import select
import termios
import tty
import threading
from datetime import datetime
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
from std_msgs.msg import Header


# ===========================================================================
# Configuracion
# ===========================================================================
RESOLUTION      = 0.05      # metros/celda
INITIAL_SIZE_M  = 40.0      # tamano inicial (crece si hace falta)
EXPAND_MARGIN_M = 15.0      # margen extra al expandir

L_FREE   = -0.4
L_OCC    =  0.85
L_PRIOR  =  0.0
L_MIN    = -5.0
L_MAX    =  5.0

MAP_PUBLISH_HZ = 4.0
MAP_FRAME      = "odom"

MAPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "maps")

PGM_FREE_THRESH = 0.25
PGM_OCC_THRESH  = 0.65

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


# ===========================================================================
# Bresenham
# ===========================================================================
def bresenham(x0, y0, x1, y1):
    cells = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1
    err = dx - dy
    while not (x0 == x1 and y0 == y1):
        cells.append((x0, y0))
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy
    return cells


# ===========================================================================
# Mapa dinamico
# ===========================================================================
class DynamicMap:
    """Grid de log-odds que crece automaticamente en cualquier direccion."""

    def __init__(self, initial_size_m, resolution):
        self._res = resolution
        half = initial_size_m / 2.0
        self._ox  = -half
        self._oy  = -half
        self._w   = int(initial_size_m / resolution)
        self._h   = int(initial_size_m / resolution)
        self._grid = np.full((self._h, self._w), L_PRIOR, dtype=np.float32)

    @property
    def origin_x(self): return self._ox
    @property
    def origin_y(self): return self._oy
    @property
    def width(self):    return self._w
    @property
    def height(self):   return self._h
    @property
    def grid(self):     return self._grid

    def world_to_cell(self, wx, wy):
        return int((wx - self._ox) / self._res), int((wy - self._oy) / self._res)

    def ensure_world_point(self, wx, wy, margin_m=0.0):
        """Expande el grid si (wx,wy)+margen no cabe. Thread-safe si se llama bajo lock."""
        margin_c = int(math.ceil(margin_m / self._res))
        cx = (wx - self._ox) / self._res
        cy = (wy - self._oy) / self._res

        pad_left  = max(0, margin_c - int(cx))
        pad_right = max(0, int(math.ceil(cx)) + margin_c - (self._w - 1))
        pad_down  = max(0, margin_c - int(cy))
        pad_up    = max(0, int(math.ceil(cy)) + margin_c - (self._h - 1))

        if pad_left == 0 and pad_right == 0 and pad_down == 0 and pad_up == 0:
            return False  # sin cambio

        new_w = self._w + pad_left + pad_right
        new_h = self._h + pad_down + pad_up
        new_grid = np.full((new_h, new_w), L_PRIOR, dtype=np.float32)
        new_grid[pad_down:pad_down + self._h,
                 pad_left:pad_left + self._w] = self._grid

        self._grid = new_grid
        self._w    = new_w
        self._h    = new_h
        self._ox  -= pad_left * self._res
        self._oy  -= pad_down * self._res
        return True  # expandido

    def update_free(self, cx, cy):
        if 0 <= cx < self._w and 0 <= cy < self._h:
            self._grid[cy, cx] = max(L_MIN, self._grid[cy, cx] + L_FREE)

    def update_occ(self, cx, cy):
        if 0 <= cx < self._w and 0 <= cy < self._h:
            self._grid[cy, cx] = min(L_MAX, self._grid[cy, cx] + L_OCC)

    def reset(self):
        half = INITIAL_SIZE_M / 2.0
        self._ox  = -half
        self._oy  = -half
        self._w   = int(INITIAL_SIZE_M / self._res)
        self._h   = int(INITIAL_SIZE_M / self._res)
        self._grid = np.full((self._h, self._w), L_PRIOR, dtype=np.float32)


# ===========================================================================
# Teclado sin bloqueo
# ===========================================================================
class KeyboardListener:
    def __init__(self):
        self._old    = None
        self._active = False

    def start(self):
        try:
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._active = True
        except Exception:
            self._active = False

    def stop(self):
        if self._old is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
            except Exception:
                pass

    def get_key(self):
        if not self._active:
            return ''
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return ''


# ===========================================================================
# Nodo principal
# ===========================================================================
class OccupancyGridMapper(Node):

    def __init__(self):
        super().__init__("occupancy_grid_mapper")

        self._map  = DynamicMap(INITIAL_SIZE_M, RESOLUTION)
        self._lock = threading.Lock()

        self._robot_x   = 0.0
        self._robot_y   = 0.0
        self._robot_yaw = 0.0
        self._have_odom = False

        self.create_subscription(Odometry,  "/odom", self._odom_cb, SENSOR_QOS)
        self.create_subscription(LaserScan, "/scan", self._scan_cb, SENSOR_QOS)

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.create_timer(1.0 / MAP_PUBLISH_HZ, self._publish_map)

        self._kb = KeyboardListener()
        self._kb.start()
        self.create_timer(0.1, self._check_keyboard)

        os.makedirs(MAPS_DIR, exist_ok=True)
        self._scan_count  = 0
        self._save_count  = 0
        self._last_expand = False
        self._last_save_time = 0.0    # antirrebote: evita guardados múltiples

        import os as _os
        self.get_logger().info(
            f"\n"
            f"  *** OccupancyGridMapper DINAMICO ***\n"
            f"  PID: {_os.getpid()}\n"
            f"  Tamano inicial: {INITIAL_SIZE_M}x{INITIAL_SIZE_M} m "
            f"= {self._map.width}x{self._map.height} celdas\n"
            f"  Margen de expansion: {EXPAND_MARGIN_M} m\n"
            f"  Resolucion: {RESOLUTION} m/celda\n"
            f"  Maps dir: {MAPS_DIR}\n"
            f"  Teclas: [m] guardar | [r] resetear | [q] salir\n"
        )

    # ------------------------------------------------------------------
    def _check_keyboard(self):
        key = self._kb.get_key()
        if key == 'm':
            if time.time() - self._last_save_time > 1.5:
                self._last_save_time = time.time()
                threading.Thread(target=self._save_map, daemon=True).start()
        elif key == 'r':
            self._reset_map()
        elif key == 'q':
            self.get_logger().info("Saliendo...")
            self._kb.stop()
            raise SystemExit

    # ------------------------------------------------------------------
    def _odom_cb(self, msg):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._have_odom = True

    # ------------------------------------------------------------------
    def _scan_cb(self, msg):
        if not self._have_odom:
            return

        rx, ry, ryaw = self._robot_x, self._robot_y, self._robot_yaw

        with self._lock:
            expanded = self._map.ensure_world_point(rx, ry, EXPAND_MARGIN_M)
            if expanded:
                self.get_logger().info(
                    f"Mapa expandido -> {self._map.width}x{self._map.height} celdas "
                    f"({self._map.width * RESOLUTION:.1f}x"
                    f"{self._map.height * RESOLUTION:.1f} m), "
                    f"origen=({self._map.origin_x:.1f},{self._map.origin_y:.1f})"
                )

            rx_c, ry_c = self._map.world_to_cell(rx, ry)
            angle = msg.angle_min + ryaw

            for r in msg.ranges:
                if msg.range_min < r < msg.range_max:
                    hx = rx + r * math.cos(angle)
                    hy = ry + r * math.sin(angle)

                    hit_expanded = self._map.ensure_world_point(hx, hy, 1.0)
                    if hit_expanded:
                        # Recalcular posicion del robot tras expansion
                        rx_c, ry_c = self._map.world_to_cell(rx, ry)

                    hx_c, hy_c = self._map.world_to_cell(hx, hy)

                    for (cx, cy) in bresenham(rx_c, ry_c, hx_c, hy_c):
                        self._map.update_free(cx, cy)
                    self._map.update_occ(hx_c, hy_c)

                angle += msg.angle_increment

            # Snapshot de stats dentro del lock (evita race condition)
            if self._scan_count % 50 == 0:
                w   = self._map.width
                h   = self._map.height
                occ  = int(np.sum(self._map.grid > 0.5))
                free = int(np.sum(self._map.grid < -0.5))
            else:
                w = h = occ = free = None

        self._scan_count += 1
        if w is not None:
            self.get_logger().info(
                f"Scans: {self._scan_count} | "
                f"Mapa: {w}x{h} ({w*RESOLUTION:.0f}x{h*RESOLUTION:.0f} m) | "
                f"Ocupadas: {occ} | Libres: {free}"
            )

    # ------------------------------------------------------------------
    def _publish_map(self):
        with self._lock:
            g  = self._map.grid.copy()
            w  = self._map.width
            h  = self._map.height
            ox = self._map.origin_x
            oy = self._map.origin_y

        prob = 1.0 / (1.0 + np.exp(-g))
        grid = np.full(g.shape, -1, dtype=np.int8)
        grid[prob < 0.35] = 0
        grid[prob > 0.65] = 100

        msg = OccupancyGrid()
        msg.header.stamp       = self.get_clock().now().to_msg()
        msg.header.frame_id    = MAP_FRAME
        msg.info.map_load_time = msg.header.stamp
        msg.info.resolution    = RESOLUTION
        msg.info.width         = w
        msg.info.height        = h
        msg.info.origin.position.x    = ox
        msg.info.origin.position.y    = oy
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.ravel().tolist()
        self.map_pub.publish(msg)

    # ------------------------------------------------------------------
    def _save_map(self):
        with self._lock:
            g  = self._map.grid.copy()
            w  = self._map.width
            h  = self._map.height
            ox = self._map.origin_x
            oy = self._map.origin_y

        self._save_count += 1
        ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"maze_map_{ts}"
        pgm_file  = os.path.join(MAPS_DIR, f"{base_name}.pgm")
        yaml_file = os.path.join(MAPS_DIR, f"{base_name}.yaml")

        prob = 1.0 / (1.0 + np.exp(-g))
        img  = np.full((h, w), 205, dtype=np.uint8)
        img[prob < PGM_FREE_THRESH] = 254
        img[prob > PGM_OCC_THRESH]  = 0

        # Recortar el PGM a la bounding box de celdas exploradas
        # (elimina el relleno de celdas desconocidas alrededor del mapa real)
        explored_mask = (img != 205)
        rows = np.any(explored_mask, axis=1)
        cols = np.any(explored_mask, axis=0)
        if rows.any():
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            # Añadir margen de 20 celdas (1 metro) alrededor
            margin = 20
            rmin = max(0, rmin - margin)
            rmax = min(h - 1, rmax + margin)
            cmin = max(0, cmin - margin)
            cmax = min(w - 1, cmax + margin)
            img_crop = img[rmin:rmax+1, cmin:cmax+1]
            # El origen del recorte en coordenadas mundo:
            # La fila 0 del grid (row index 0) corresponde a oy en mundo.
            # Pero el PGM tiene fila 0 = arriba (y mayor en mundo).
            # origin_x del recorte = ox + cmin * RESOLUTION
            # origin_y del recorte = oy + rmin * RESOLUTION
            crop_ox = ox + cmin * RESOLUTION
            crop_oy = oy + rmin * RESOLUTION
            save_w = img_crop.shape[1]
            save_h = img_crop.shape[0]
        else:
            # Sin datos explorados: guardar el grid completo
            img_crop = img
            crop_ox, crop_oy = ox, oy
            save_w, save_h = w, h

        with open(pgm_file, 'wb') as f:
            f.write(f"P5\n{save_w} {save_h}\n255\n".encode('ascii'))
            f.write(np.flipud(img_crop).tobytes())

        with open(yaml_file, 'w') as f:
            f.write(
                f"image: {base_name}.pgm\n"
                f"mode: trinary\n"
                f"resolution: {RESOLUTION}\n"
                f"origin: [{crop_ox}, {crop_oy}, 0.0]\n"
                f"negate: 0\n"
                f"occupied_thresh: {PGM_OCC_THRESH}\n"
                f"free_thresh: {PGM_FREE_THRESH}\n"
            )

        occ  = int(np.sum(prob > PGM_OCC_THRESH))
        free = int(np.sum(prob < PGM_FREE_THRESH))
        self.get_logger().info(
            f"\n  ========================================\n"
            f"  MAPA GUARDADO #{self._save_count}\n"
            f"  Grid total:   {w}x{h} celdas ({w*RESOLUTION:.1f}x{h*RESOLUTION:.1f} m)\n"
            f"  Recortado a:  {save_w}x{save_h} celdas ({save_w*RESOLUTION:.1f}x{save_h*RESOLUTION:.1f} m)\n"
            f"  Origen:       ({crop_ox:.2f}, {crop_oy:.2f})\n"
            f"  PGM:  {pgm_file}\n"
            f"  YAML: {yaml_file}\n"
            f"  Ocupadas: {occ} | Libres: {free} | Scans: {self._scan_count}\n"
            f"  ========================================"
        )

    # ------------------------------------------------------------------
    def _reset_map(self):
        with self._lock:
            self._map.reset()
        self._scan_count = 0
        self.get_logger().info("Mapa reseteado.")

    def destroy_node(self):
        self._kb.stop()
        super().destroy_node()


# ===========================================================================
# main
# ===========================================================================
def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridMapper()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
