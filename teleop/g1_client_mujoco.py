#!/usr/bin/env python3
"""
g1_client_mujoco.py
===================
Cliente de teleoperación para el robot G1 en simulación MuJoCo.

Características:
  - Feed de cámara en vivo (ZMQ puerto 5555) con overlay de estado
  - Control WASD + QE por teclado — multi-tecla combinable
  - Joystick virtual en pantalla (click + arrastrar)
  - Panel de telemetría: teclas activas, conexión, FPS

Conexiones:
  - ZMQ SUB tcp://127.0.0.1:5555  → frames de cámara JPEG
  - TCP 127.0.0.1:6000             → comandos (conexión persistente, JSON {"keys":[...]})
  - UDP 127.0.0.1:6005             → reset de posición

Uso:
    python3 teleop/g1_client_mujoco.py

Controles:
    W / S     Avanzar / Retroceder
    A / D     Strafe izquierda / derecha
    Q / E     Rotar izquierda / derecha
    ESPACIO   Stop de emergencia
    R         Resetear posición del robot
    ESC       Cerrar
"""

import json
import socket
import threading
import time
import os
import math

import zmq
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# ============================================================
# Configuración
# ============================================================
ROBOT_IP   = '127.0.0.1'
CAM_PORT   = "5555"
CMD_PORT   = 6000
RESET_PORT = 6005

TARGET_FPS = 30
FRAME_MS   = int(1000 / TARGET_FPS)

WIN_W = 940
WIN_H = 680
CAM_W = 640
CAM_H = 480
PANEL_W = WIN_W - CAM_W
PANEL_X = CAM_W

JOY_R  = 68
JOY_CX = PANEL_X + PANEL_W // 2
JOY_CY = 490

# Colores BGR
C_BG     = (14, 14, 16)
C_PANEL  = (20, 20, 24)
C_BORDER = (42, 46, 52)
C_WHITE  = (235, 235, 240)
C_GRAY   = (100, 105, 115)
C_GREEN  = (50, 210, 90)
C_RED    = (50,  50, 210)
C_YELLOW = (30, 200, 220)
C_ACCENT = (200, 170,  40)
C_ACCENT2= (180, 100,  20)

MOVE_KEYS = {'w', 'a', 's', 'd', 'q', 'e'}
KEY_LABELS = {
    'w': 'FWD', 's': 'REV', 'a': 'STR-L',
    'd': 'STR-R', 'q': 'ROT-L', 'e': 'ROT-R',
}

# ============================================================
# Utilidades de dibujo
# ============================================================
FONT = cv2.FONT_HERSHEY_DUPLEX

def txt(img, text, x, y, scale=0.48, color=C_WHITE, thick=1):
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

def txc(img, text, cx, cy, scale=0.48, color=C_WHITE, thick=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (cx - tw // 2, cy + th // 2), FONT, scale, color, thick, cv2.LINE_AA)

def rrect(img, x1, y1, x2, y2, r, color, thick=-1):
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thick)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thick)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, thick)

def hline(img, x1, x2, y, color=C_BORDER):
    cv2.line(img, (x1, y), (x2, y), color, 1)

def semi(img, x1, y1, x2, y2, color, alpha=0.55):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


# ============================================================
# Aplicación principal
# ============================================================
class G1TeleopClient:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("G1 Teleop")
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.configure(bg="#0e0e10")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=WIN_W, height=WIN_H,
                                bg="#0e0e10", highlightthickness=0)
        self.canvas.pack()
        self._tk_img = None

        # ── Estado multi-tecla ────────────────────────────────
        self._keys_down  = set()
        self._stop_timer = None
        self._send_timer = None
        self._RESEND_MS  = 80   # reenvío mientras hay teclas pulsadas

        # ── Joystick ──────────────────────────────────────────
        self._joy_active = False
        self._joy_dx     = 0.0
        self._joy_dy     = 0.0

        # ── Telemetría ────────────────────────────────────────
        self._fps_counter = 0
        self._fps_display = 0
        self._last_fps_t  = time.time()
        self._frame_ok    = False
        self._conn_ok     = False
        self._cmd_label   = 'IDLE'
        self._cmd_color   = C_GRAY

        # ── ZMQ cámara ────────────────────────────────────────
        self.ctx      = zmq.Context()
        self.sock_cam = self.ctx.socket(zmq.SUB)
        self.sock_cam.setsockopt(zmq.CONFLATE, 1)
        self.sock_cam.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock_cam.connect(f"tcp://{ROBOT_IP}:{CAM_PORT}")

        self._last_frame = self._make_no_signal_frame()
        self._alive = True

        # ── Conexión TCP persistente ──────────────────────────
        # Una sola conexión que se mantiene abierta y manda JSON
        # separados por \n. El servidor ya soporta este modo.
        self._tcp_sock = None
        self._tcp_lock = threading.Lock()
        self._connect_tcp()

        # ── Bindings ──────────────────────────────────────────
        root.bind('<KeyPress>',    self._on_key_press)
        root.bind('<KeyRelease>',  self._on_key_release)
        root.focus_set()

        self.canvas.bind('<ButtonPress-1>',   self._joy_press)
        self.canvas.bind('<B1-Motion>',       self._joy_motion)
        self.canvas.bind('<ButtonRelease-1>', self._joy_release)

        # ── Hilos ─────────────────────────────────────────────
        threading.Thread(target=self._cam_loop,   daemon=True).start()
        threading.Thread(target=self._reconnect_loop, daemon=True).start()

        self._render_loop()

    # ════════════════════════════════════════════════════════
    # CONEXIÓN TCP PERSISTENTE
    # ════════════════════════════════════════════════════════
    def _connect_tcp(self):
        """Intenta abrir la conexión persistente al servidor."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((ROBOT_IP, CMD_PORT))
            s.settimeout(None)   # sin timeout en modo persistente
            with self._tcp_lock:
                self._tcp_sock = s
            self._conn_ok = True
        except Exception:
            with self._tcp_lock:
                self._tcp_sock = None
            self._conn_ok = False

    def _reconnect_loop(self):
        """Hilo que vigila la conexión y reconecta si se pierde."""
        while self._alive:
            time.sleep(1.0)
            with self._tcp_lock:
                alive = self._tcp_sock is not None
            if not alive:
                self._connect_tcp()
            else:
                # Ping ligero: enviar "ping\n" y leer respuesta
                ok = self._send_raw('ping')
                self._conn_ok = ok
                if not ok:
                    with self._tcp_lock:
                        try:
                            self._tcp_sock.close()
                        except Exception:
                            pass
                        self._tcp_sock = None

    def _send_raw(self, msg: str) -> bool:
        """Envía msg+\\n por la conexión persistente. Hilo-seguro."""
        payload = (msg if msg.endswith('\n') else msg + '\n').encode()
        with self._tcp_lock:
            sock = self._tcp_sock
        if sock is None:
            return False
        try:
            sock.sendall(payload)
            return True
        except Exception:
            # Conexión rota — marcar para reconexión
            with self._tcp_lock:
                self._tcp_sock = None
            self._conn_ok = False
            return False

    # ════════════════════════════════════════════════════════
    # CÁMARA
    # ════════════════════════════════════════════════════════
    def _cam_loop(self):
        while self._alive:
            try:
                raw = self.sock_cam.recv(zmq.NOBLOCK)
                arr = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self._last_frame = frame
                    self._frame_ok   = True
                    self._fps_counter += 1
            except zmq.Again:
                time.sleep(0.005)
            except Exception:
                time.sleep(0.01)

    # ════════════════════════════════════════════════════════
    # TECLADO — multi-tecla
    # ════════════════════════════════════════════════════════
    def _on_key_press(self, event):
        key = event.keysym.lower()

        if key == 'escape':
            self.close(); self.root.destroy(); os._exit(0)
        if key == 'space':
            self._emergency_stop(); return
        if key == 'r':
            self._send_reset(); return
        if key not in MOVE_KEYS:
            return

        if self._stop_timer is not None:
            self.root.after_cancel(self._stop_timer)
            self._stop_timer = None

        if key not in self._keys_down:
            self._keys_down.add(key)
            self._schedule_resend()

    def _on_key_release(self, event):
        key = event.keysym.lower()
        if key not in MOVE_KEYS:
            return
        self._keys_down.discard(key)

        if self._keys_down:
            self._schedule_resend()
        else:
            if self._send_timer is not None:
                self.root.after_cancel(self._send_timer)
                self._send_timer = None
            if not self._joy_active:
                self._stop_timer = self.root.after(40, self._do_stop)

    def _schedule_resend(self):
        if self._send_timer is not None:
            self.root.after_cancel(self._send_timer)
        self._send_keys_now()

    def _send_keys_now(self):
        if not self._alive:
            return
        keys = list(self._keys_down)
        if keys:
            self._dispatch_keys(keys)
            self._send_timer = self.root.after(self._RESEND_MS, self._send_keys_now)
        else:
            self._send_timer = None

    def _emergency_stop(self):
        self._keys_down.clear()
        self._joy_active = False
        self._joy_dx = self._joy_dy = 0.0
        for t in (self._stop_timer, self._send_timer):
            if t is not None:
                self.root.after_cancel(t)
        self._stop_timer = self._send_timer = None
        self._send_raw('stop')
        self._cmd_label = '!! STOP'
        self._cmd_color = C_RED

    def _do_stop(self):
        self._stop_timer = None
        if not self._keys_down and not self._joy_active:
            self._send_raw('stop')
            self._cmd_label = 'IDLE'
            self._cmd_color = C_GRAY

    def _dispatch_keys(self, keys):
        parts = [KEY_LABELS[k] for k in keys if k in KEY_LABELS]
        self._cmd_label = '+'.join(parts) if parts else 'MOVE'
        self._cmd_color = C_ACCENT
        self._send_raw(json.dumps({"keys": keys}))

    # ════════════════════════════════════════════════════════
    # JOYSTICK VIRTUAL
    # ════════════════════════════════════════════════════════
    def _joy_in_zone(self, ex, ey):
        return math.hypot(ex - JOY_CX, ey - JOY_CY) < JOY_R * 2.2

    def _joy_press(self, event):
        if self._joy_in_zone(event.x, event.y):
            self._joy_active = True
            self._update_joy(event.x, event.y)

    def _joy_motion(self, event):
        if self._joy_active:
            self._update_joy(event.x, event.y)

    def _joy_release(self, event):
        if self._joy_active:
            self._joy_active = False
            self._joy_dx = self._joy_dy = 0.0
            self._do_stop()

    def _update_joy(self, mx, my):
        dx = mx - JOY_CX
        dy = my - JOY_CY
        dist = math.hypot(dx, dy)
        if dist > JOY_R:
            dx = dx / dist * JOY_R
            dy = dy / dist * JOY_R
        self._joy_dx =  dx / JOY_R
        self._joy_dy = -dy / JOY_R

        keys = []
        if self._joy_dy >  0.2: keys.append('w')
        if self._joy_dy < -0.2: keys.append('s')
        if self._joy_dx < -0.2: keys.append('a')
        if self._joy_dx >  0.2: keys.append('d')

        if keys:
            self._dispatch_keys(keys)
        else:
            self._do_stop()

    # ════════════════════════════════════════════════════════
    # RED — solo la conexión persistente
    # ════════════════════════════════════════════════════════
    def _send_reset(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(b"reset", (ROBOT_IP, RESET_PORT))
            s.close()
        except Exception:
            pass

    # ════════════════════════════════════════════════════════
    # RENDER
    # ════════════════════════════════════════════════════════
    def _render_loop(self):
        if not self._alive:
            return
        now = time.time()
        if now - self._last_fps_t >= 1.0:
            self._fps_display = self._fps_counter
            self._fps_counter = 0
            self._last_fps_t  = now

        img = self._build_frame()
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._tk_img)
        self.root.after(FRAME_MS, self._render_loop)

    # ════════════════════════════════════════════════════════
    # CONSTRUCCIÓN DEL FRAME
    # ════════════════════════════════════════════════════════
    def _build_frame(self):
        img = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        img[:] = C_BG

        cam_y = (WIN_H - CAM_H) // 2
        frame = cv2.resize(self._last_frame, (CAM_W, CAM_H))
        img[cam_y:cam_y + CAM_H, 0:CAM_W] = frame

        bc = C_GREEN if self._frame_ok else (55, 55, 55)
        cv2.rectangle(img, (0, cam_y), (CAM_W - 1, cam_y + CAM_H - 1), bc, 2)

        self._draw_cam_overlay(img, cam_y)
        cv2.rectangle(img, (CAM_W, 0), (CAM_W + 2, WIN_H), C_BORDER, -1)
        cv2.rectangle(img, (PANEL_X + 2, 0), (WIN_W - 1, WIN_H - 1), C_PANEL, -1)
        self._draw_panel(img)
        return img

    def _draw_cam_overlay(self, img, cam_y):
        semi(img, 0, cam_y, CAM_W, cam_y + 30, (0, 0, 0), 0.60)
        txt(img, "G1  LIVE FEED", 8, cam_y + 20, 0.55, C_WHITE)
        fps_c = C_GREEN if self._fps_display >= 20 else C_YELLOW
        txt(img, f"{self._fps_display} fps", CAM_W - 72, cam_y + 20, 0.48, fps_c)

        semi(img, 0, cam_y + CAM_H - 28, CAM_W, cam_y + CAM_H, (0, 0, 0), 0.60)
        dot_c = C_GREEN if self._conn_ok else (60, 60, 60)
        cv2.circle(img, (10, cam_y + CAM_H - 14), 5, dot_c, -1)
        conn_t = "SIM CONNECTED" if self._conn_ok else "SIM OFFLINE"
        txt(img, conn_t, 20, cam_y + CAM_H - 8, 0.42, dot_c)
        cam_t = "CAM OK" if self._frame_ok else "NO SIGNAL"
        cam_c = C_GREEN if self._frame_ok else (65, 65, 65)
        txt(img, cam_t, CAM_W - 88, cam_y + CAM_H - 8, 0.42, cam_c)

        if self._cmd_label not in ('IDLE', 'ping'):
            label = self._cmd_label
            lw = max(150, len(label) * 16 + 24)
            bx = (CAM_W - lw) // 2
            by = cam_y + CAM_H - 70
            semi(img, bx - 4, by - 4, bx + lw + 4, by + 32, (0, 0, 0), 0.65)
            rrect(img, bx, by, bx + lw, by + 28, 5, self._cmd_color, 2)
            txc(img, label, bx + lw // 2, by + 14, 0.65, self._cmd_color, 2)

    def _draw_panel(self, img):
        px = PANEL_X + 8
        pw = PANEL_W - 16

        txc(img, "TELEOP", PANEL_X + PANEL_W // 2, 18, 0.6, C_ACCENT, 1)
        hline(img, PANEL_X + 4, WIN_W - 4, 32)

        self._draw_wasd(img, PANEL_X + PANEL_W // 2, 80)
        hline(img, PANEL_X + 4, WIN_W - 4, 190)

        self._draw_cmd_status(img, px, 204, pw)
        hline(img, PANEL_X + 4, WIN_W - 4, 270)

        self._draw_joystick(img)
        hline(img, PANEL_X + 4, WIN_W - 4, 570)

        self._draw_buttons(img, px, 582, pw)
        self._draw_legend(img, px, WIN_H - 68)

    def _draw_wasd(self, img, cx, top):
        ks  = 38
        gap = 5
        layout = [
            ('q', -1, 0), ('w', 0, 0), ('e', 1, 0),
            ('a', -1, 1), ('s', 0, 1), ('d', 1, 1),
        ]
        sublabels = {
            'w': '^', 's': 'v', 'a': '<', 'd': '>', 'q': 'CCW', 'e': 'CW',
        }
        for k, col, row in layout:
            kx = cx + col * (ks + gap) - ks // 2
            ky = top + row * (ks + gap)
            active = k in self._keys_down
            bg     = C_ACCENT  if active else (30, 32, 38)
            border = C_ACCENT  if active else (55, 60, 70)
            tc     = (12, 12, 12) if active else (140, 145, 155)
            sub_c  = (12, 12, 12) if active else (70, 75, 85)
            rrect(img, kx, ky, kx + ks, ky + ks, 6, bg)
            rrect(img, kx, ky, kx + ks, ky + ks, 6, border, 1)
            txc(img, k.upper(), kx + ks // 2, ky + ks // 2 - 5, 0.6, tc, 1)
            txc(img, sublabels[k], kx + ks // 2, ky + ks - 7, 0.28, sub_c, 1)

        sx  = cx - (ks + gap) - ks // 2
        sy  = top + 2 * (ks + gap) + 6
        sw  = 3 * ks + 2 * gap
        rrect(img, sx, sy, sx + sw, sy + 18, 4, (28, 18, 18))
        rrect(img, sx, sy, sx + sw, sy + 18, 4, (80, 28, 28), 1)
        txc(img, "SPACE  =  E-STOP", sx + sw // 2, sy + 9,
            0.32, (100, 35, 35))

    def _draw_cmd_status(self, img, px, y, pw):
        txc(img, "ACTIVE KEYS", PANEL_X + PANEL_W // 2, y + 8, 0.38, C_GRAY)
        keys = list(self._keys_down)
        if not keys and not self._joy_active:
            txc(img, "-  IDLE  -", PANEL_X + PANEL_W // 2, y + 36, 0.5, C_GRAY)
        else:
            bw = 40
            total = len(keys) * (bw + 6) - 6
            bx = PANEL_X + PANEL_W // 2 - total // 2
            for k in keys:
                rrect(img, bx, y + 20, bx + bw, y + 52, 5, C_ACCENT)
                txc(img, k.upper(), bx + bw // 2, y + 36, 0.55, (10, 10, 10), 1)
                bx += bw + 6
            if self._joy_active:
                txc(img, "JOY", bx + 20, y + 36, 0.5, C_ACCENT2, 1)

    def _draw_joystick(self, img):
        cx, cy, r = JOY_CX, JOY_CY, JOY_R
        txc(img, "VIRTUAL JOYSTICK", cx, 582 - 14, 0.36, C_GRAY)
        txc(img, "click + drag", cx, 582 - 2, 0.30, (60, 65, 75))

        cv2.circle(img, (cx, cy), r, (24, 26, 32), -1)
        cv2.circle(img, (cx, cy), r, C_BORDER, 2)
        cv2.line(img, (cx - r, cy), (cx + r, cy), (38, 42, 50), 1)
        cv2.line(img, (cx, cy - r), (cx, cy + r), (38, 42, 50), 1)
        for ri in [r // 3, 2 * r // 3]:
            cv2.circle(img, (cx, cy), ri, (32, 36, 44), 1)

        txt(img, "^", cx - 5, cy - r - 6,  0.38, (60, 65, 75))
        txt(img, "v", cx - 4, cy + r + 14, 0.38, (60, 65, 75))
        txt(img, "<", cx - r - 14, cy + 5, 0.38, (60, 65, 75))
        txt(img, ">", cx + r + 4,  cy + 5, 0.38, (60, 65, 75))

        if self._joy_active:
            jx = int(cx + self._joy_dx * r)
            jy = int(cy - self._joy_dy * r)
            cv2.line(img, (cx, cy), (jx, jy), C_ACCENT, 1)
            cv2.circle(img, (jx, jy), 14, C_ACCENT, -1)
            cv2.circle(img, (jx, jy), 14, C_WHITE, 1)
        else:
            cv2.circle(img, (cx, cy), 14, (44, 48, 58), -1)
            cv2.circle(img, (cx, cy), 14, (70, 76, 88), 2)

    def _draw_buttons(self, img, px, y, pw):
        rrect(img, px, y, px + pw, y + 28, 5, (80, 16, 16))
        rrect(img, px, y, px + pw, y + 28, 5, (160, 36, 36), 1)
        txc(img, "E-STOP  [SPACE]", px + pw // 2, y + 14, 0.46, (220, 70, 70))

        rrect(img, px, y + 34, px + pw, y + 62, 5, (14, 32, 65))
        rrect(img, px, y + 34, px + pw, y + 62, 5, (30, 70, 150), 1)
        txc(img, "RESET ROBOT  [R]", px + pw // 2, y + 48, 0.42, (70, 130, 210))

    def _draw_legend(self, img, px, y):
        hline(img, PANEL_X + 4, WIN_W - 4, y - 4)
        items = [
            ("W / S", "Forward / Backward"),
            ("A / D", "Strafe Left / Right"),
            ("Q / E", "Rotate Left / Right"),
            ("SPACE", "Emergency Stop"),
            ("R",     "Reset Robot"),
        ]
        col2_x = px + 52
        for i, (key, desc) in enumerate(items):
            ly = y + 4 + i * 14
            txt(img, key, px,     ly, 0.30, C_ACCENT)
            txt(img, desc, col2_x, ly, 0.30, (95, 100, 110))

    # ════════════════════════════════════════════════════════
    # NO SIGNAL FRAME
    # ════════════════════════════════════════════════════════
    def _make_no_signal_frame(self):
        img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        img[:] = (14, 14, 18)
        noise = np.random.randint(0, 18, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        txc(img, "NO SIGNAL",               CAM_W // 2, CAM_H // 2 - 18, 1.1, (50, 55, 65), 2)
        txc(img, "Waiting for simulator...", CAM_W // 2, CAM_H // 2 + 18, 0.5, (42, 46, 55))
        return img

    # ════════════════════════════════════════════════════════
    # CIERRE
    # ════════════════════════════════════════════════════════
    def close(self):
        self._alive = False
        with self._tcp_lock:
            if self._tcp_sock:
                try: self._tcp_sock.close()
                except Exception: pass
                self._tcp_sock = None
        try:
            self.sock_cam.close()
            self.ctx.term()
        except Exception:
            pass


# ============================================================
# Entrypoint
# ============================================================
def main():
    root = tk.Tk()
    app  = G1TeleopClient(root)

    def on_close():
        app.close()
        root.destroy()
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    import signal
    signal.signal(signal.SIGINT, lambda *_: on_close())
    root.mainloop()


if __name__ == "__main__":
    main()
