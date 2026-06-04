"""
vision_brain_real.py
====================
Brain para el robot real Unitree G1 (29 DOF).

Integra directamente:
  • Cámara RGB del robot  (Unitree SDK VideoClient)
  • Cámara de profundidad  (Intel RealSense D435i – solo depth)
  • IA de detección        (YOLO best.pt)
  • IK bimanual            (Pinocchio, mismo motor que inverse_kinematics.py)
  • Publicación de comandos (ROS2 / unitree_hg LowCmd, misma interfaz que IK)

Geometría de la cámara (robot real G1)
---------------------------------------
  Origen del robot   : base del robot (punto naranja), coordenadas (X=adelante, Y=izquierda, Z=arriba)
  Offset de la cámara: +47.64 mm en X (hacia adelante), +462.68 mm en Z (altura)
                       La cámara está ligeramente a la derecha del centro → desplazamiento -Y pequeño
                       (ver CAMERA_OFFSET_Y abajo, ajustar si es necesario)
  Inclinación        : 42.4° hacia abajo desde la vertical
                       → el eje Z_cam (hacia la escena) apunta: X_robot +sin(42.4°), Z_robot -cos(42.4°)

Ejes del frame óptico (RealSense estándar):
  X_cam = derecha  → en robot es -Y_robot
  Y_cam = abajo    → en robot es -Z_robot  (con rotación de inclinación incluida)
  Z_cam = adelante → en robot es una mezcla de +X_robot / -Z_robot según el tilt

La función transform_camera_to_base() implementa la cadena completa:
  1. Rota el punto del frame de cámara inclinada al frame de cámara sin inclinar.
  2. Aplica el offset traslacional.

NO se incluye la lógica de caminar. Se asume que el robot está ya a distancia
de agarre y la caja siempre es alcanzable con los brazos.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTACIONES (mismo orden que vision_brain_test.py para cargar YOLO primero)
# ──────────────────────────────────────────────────────────────────────────────
from ultralytics import YOLO
import torch

import os, sys, time, threading, math
import numpy as np
import pinocchio as pin
import cv2
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from unitree_hg.msg import LowCmd, LowState

# Unitree SDK para la cámara de color
sdk_path = "/root/unitree_sdk2_python"
if sdk_path not in sys.path:
    sys.path.append(sdk_path)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PARÁMETROS GLOBALES
# ──────────────────────────────────────────────────────────────────────────────

# Cámara (depth 640×480, mismo que vision_brain_test.py)
CX             = 320.0
CY             = 240.0
FOCAL_LENGTH   = 460.0
IMG_W, IMG_H   = 640, 480

# Geometría de montaje de la cámara en el robot real
CAMERA_TILT_DEG   = 42.4                          # grados hacia abajo desde la vertical
CAMERA_TILT_RAD   = math.radians(CAMERA_TILT_DEG)
# Offset [X, Y, Z] de la cámara respecto al origen robot (en metros)
# X=adelante, Y=izquierda, Z=arriba   (G1 convención)
CAMERA_OFFSET_X   =  0.04764   # 47.64 mm hacia adelante
CAMERA_OFFSET_Y   = -0.020     # ~20 mm hacia la derecha del centro (ajustar si hace falta)
CAMERA_OFFSET_Z   =  0.46268   # 462.68 mm de altura

# Caja real
ANCHO_CAJA_REAL       = 0.20   # m   (eje Y del robot, lateral)
PROFUNDIDAD_CAJA_REAL = 0.20   # m   (eje X del robot, adelante)
LONGITUD_MANO         = 0.12   # m   distancia muñeca → palma/dedos

# YOLO
YOLO_MODEL_PATH = "best.pt"
YOLO_CONF       = 0.25

# Índices de articulaciones G1 (igual que inverse_kinematics.py / vision_brain.py)
G1_ARM_LEFT    = [15, 16, 17, 18, 19, 20, 21]
G1_ARM_RIGHT   = [22, 23, 24, 25, 26, 27, 28]
G1_WAIST       = [12, 13, 14]
NOT_USED_JOINT = 29      # token de habilitación del /arm_sdk

# IK
KP = 60.0
KD = 1.5
DT = 0.02   # 50 Hz

# URDF
URDF_PATH = os.path.expanduser(
    "~/robot_ws/src/g1pilot/description_files/urdf/g1_29dof.urdf")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  TRANSFORMACIÓN CÁMARA → BASE ROBOT  (la pieza clave)
# ──────────────────────────────────────────────────────────────────────────────

def transform_camera_to_base(z_dist, x_lateral, y_vertical):
    """
    Convierte un punto 3-D expresado en el frame óptico de la cámara inclinada
    a coordenadas del frame base del robot.

    Parámetros
    ----------
    z_dist     : profundidad en línea recta (eje Z_cam = hacia la escena), metros
    x_lateral  : desplazamiento lateral   (eje X_cam = derecha), metros
                 Fórmula de proyección: x_lateral = ((u - CX) * z_dist) / FOCAL_LENGTH
    y_vertical : desplazamiento vertical  (eje Y_cam = abajo),   metros
                 Fórmula de proyección: y_vertical = ((v - CY) * z_dist) / FOCAL_LENGTH

    Retorna
    -------
    np.array([X_robot, Y_robot, Z_robot])
        X_robot  = adelante  (positivo hacia donde mira el robot)
        Y_robot  = izquierda (positivo hacia la izquierda)
        Z_robot  = arriba    (positivo hacia arriba)

    Matemática
    ----------
    El frame óptico de la cámara tiene el eje Z apuntando hacia la escena.
    Cuando la cámara está inclinada θ = 42.4° hacia abajo desde la vertical,
    el eje Z_cam en coordenadas robot es:
        Z_cam_robot = [ sin(θ),  0, -cos(θ) ]   (adelante y abajo)
    El eje Y_cam (abajo en imagen) en robot:
        Y_cam_robot = [ cos(θ),  0,  sin(θ) ]   (adelante y arriba, por la inclinación)
    El eje X_cam (derecha en imagen) en robot:
        X_cam_robot = [    0,   -1,      0  ]   (negativo porque izq. es +Y robot)

    Por tanto un punto p = (z_dist, x_lateral, y_vertical) en cam se convierte a:
        p_cam_vec = z_dist * Z_cam_robot
                  + x_lateral * X_cam_robot
                  + y_vertical * Y_cam_robot

    Luego se suma el offset de montaje.
    """
    st = math.sin(CAMERA_TILT_RAD)  # sin(42.4°) ≈ 0.675
    ct = math.cos(CAMERA_TILT_RAD)  # cos(42.4°) ≈ 0.739

    # Contribución de cada componente en ejes robot
    # eje Z_cam → ( sin(θ)*Z,  0,          -cos(θ)*Z )
    # eje X_cam → ( 0,         -X_lat,      0        )   (X_cam = -Y_robot)
    # eje Y_cam → ( cos(θ)*Y,  0,           sin(θ)*Y )   (Y_cam gira con el tilt)
    x_robot = CAMERA_OFFSET_X + st * z_dist + ct * y_vertical
    y_robot = CAMERA_OFFSET_Y - x_lateral
    z_robot = CAMERA_OFFSET_Z - ct * z_dist + st * y_vertical

    return np.array([x_robot, y_robot, z_robot])


def project_base_to_camera(pos_base):
    """
    Proyección inversa: de coordenadas base del robot a (u, v) en imagen.
    Útil para el HUD de depuración.
    Devuelve (None, None) si el punto está detrás de la cámara.
    """
    st = math.sin(CAMERA_TILT_RAD)
    ct = math.cos(CAMERA_TILT_RAD)

    dx = pos_base[0] - CAMERA_OFFSET_X
    dy = pos_base[1] - CAMERA_OFFSET_Y
    dz = pos_base[2] - CAMERA_OFFSET_Z

    # Invertimos las ecuaciones de transform_camera_to_base
    # z_dist     =  st*dx - ct*dz
    # x_lateral  = -dy
    # y_vertical =  ct*dx + st*dz
    z_c = st * dx - ct * dz
    x_l = -dy
    y_v = ct * dx + st * dz

    if z_c < 0.05:
        return None, None

    u = int(np.clip(CX + (x_l * FOCAL_LENGTH) / z_c, 0, IMG_W - 1))
    v = int(np.clip(CY + (y_v * FOCAL_LENGTH) / z_c, 0, IMG_H - 1))
    return u, v


# ──────────────────────────────────────────────────────────────────────────────
# 3.  ESCANEADO DE MESA  (para determinar Z límite seguro)
# ──────────────────────────────────────────────────────────────────────────────

def scan_table_around_box(depth_frame, bbox):
    """
    Muestrea el mapa de profundidad alrededor de la bbox de la caja para
    estimar la profundidad de la superficie de la mesa (z_cam).
    Retorna (z_cam_mesa, n_muestras).
    """
    if depth_frame is None:
        return None, 0

    x1_d, y1_d, x2_d, y2_d = bbox
    w_d = x2_d - x1_d
    h_d = y2_d - y1_d
    box_z_cam = depth_frame[
        int(np.clip((y1_d + y2_d) // 2, 0, IMG_H - 1)),
        int(np.clip((x1_d + x2_d) // 2, 0, IMG_W - 1))
    ]

    samples = []

    # Zona justo debajo de la caja
    for row_off in range(5, 90, 3):
        row = int(np.clip(y2_d + row_off, 0, IMG_H - 1))
        for col_off in range(-w_d // 2, int(w_d * 1.5), 4):
            col = int(np.clip(x1_d + col_off, 0, IMG_W - 1))
            d = depth_frame[row, col]
            if 0.05 < d < box_z_cam * 1.6:
                samples.append(d)

    # Zona lateral izquierda
    for col_off in range(w_d // 2 + 5, w_d + 100, 4):
        col = int(np.clip(x1_d - col_off, 0, IMG_W - 1))
        for row_off in range(-h_d // 3, h_d + 30, 4):
            row = int(np.clip(y1_d + row_off, 0, IMG_H - 1))
            d = depth_frame[row, col]
            if 0.05 < d < box_z_cam * 1.6:
                samples.append(d)

    # Zona lateral derecha
    for col_off in range(w_d // 2 + 5, w_d + 100, 4):
        col = int(np.clip(x2_d + col_off, 0, IMG_W - 1))
        for row_off in range(-h_d // 3, h_d + 30, 4):
            row = int(np.clip(y1_d + row_off, 0, IMG_H - 1))
            d = depth_frame[row, col]
            if 0.05 < d < box_z_cam * 1.6:
                samples.append(d)

    if len(samples) < 15:
        return box_z_cam, len(samples)

    return float(np.percentile(samples, 15)), len(samples)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  MÓDULO DE VISIÓN  (hilo independiente que actualiza datos de la caja)
# ──────────────────────────────────────────────────────────────────────────────

class VisionModule:
    """
    Captura sincronizada de:
      - frame RGB      (Unitree VideoClient)
      - frame Depth    (RealSense D435i)
    Detección con YOLO y cálculo de posición 3-D de la arista superior de la caja.
    Corre en un hilo y expone los resultados con un lock para lectura segura.
    """
    def __init__(self):
        self.lock       = threading.Lock()
        self.running    = False

        # Resultados públicos (protegidos por self.lock)
        self.box_detected       = False
        self.box_center_base    = None   # np.array([X, Y, Z]) en frame robot
        self.z_mesa_base        = None   # Z base de la superficie de la mesa
        self.box_width_cam      = 0.0    # ancho de la caja en metros (estimado desde depth)
        self.box_height_cam     = 0.0    # alto de la caja en metros
        self.edge_x_left_base   = None   # extremo izquierdo de la arista (frame robot)
        self.edge_x_right_base  = None   # extremo derecho de la arista (frame robot)
        self.depth_frame_latest = None   # para debug externo
        self.annotated_frame    = None   # para mostrar en pantalla

        # Cámara RGB (Unitree)
        ChannelFactoryInitialize(0, "eth0")
        self.video = VideoClient()
        self.video.Init()

        # Cámara Depth (RealSense)
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
        self.pipeline.start(cfg)
        print("[VISIÓN] RealSense depth conectado.")

        # IA
        self.model = YOLO(YOLO_MODEL_PATH)
        print("[VISIÓN] Modelo YOLO cargado.")

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False
        self.pipeline.stop()

    def _loop(self):
        while self.running:
            t0 = time.time()
            try:
                ret, data = self.video.GetImageSample()
                frames = self.pipeline.wait_for_frames()
                depth_frame_rs = frames.get_depth_frame()

                if ret != 0 or not data or not depth_frame_rs:
                    time.sleep(0.01)
                    continue

                # Decodificar RGB
                np_arr = np.frombuffer(bytes(data), np.uint8)
                color_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if color_img is None:
                    time.sleep(0.01)
                    continue

                # Extraer depth como array float32
                depth_arr = np.asanyarray(depth_frame_rs.get_data()).astype(np.float32) * \
                            depth_frame_rs.get_units()   # convierte a metros

                # Escalar factores RGB → depth (640×480)
                h_c, w_c = color_img.shape[:2]
                sx = IMG_W / w_c
                sy = IMG_H / h_c

                # YOLO
                results = self.model.predict(color_img, conf=YOLO_CONF, verbose=False)
                annotated = results[0].plot()

                detected    = False
                center_base = None
                z_mesa      = None
                width_m     = 0.0
                height_m    = 0.0
                edge_l      = None
                edge_r      = None

                if len(results[0].boxes) > 0:
                    # Tomamos la detección con mayor confianza
                    best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                    x1_c, y1_c, x2_c, y2_c = map(int, best_box.xyxy[0])

                    # Escalar a coordenadas del mapa de profundidad
                    x1_d = int(x1_c * sx);  y1_d = int(y1_c * sy)
                    x2_d = int(x2_c * sx);  y2_d = int(y2_c * sy)
                    x1_d = int(np.clip(x1_d, 0, IMG_W - 1))
                    y1_d = int(np.clip(y1_d, 0, IMG_H - 1))
                    x2_d = int(np.clip(x2_d, 0, IMG_W - 1))
                    y2_d = int(np.clip(y2_d, 0, IMG_H - 1))

                    cx_d = (x1_d + x2_d) // 2
                    cy_d = (y1_d + y2_d) // 2

                    # ── Profundidad del centro de la caja ───────────────────────
                    # Mediana en ventana 11×11 alrededor del centro
                    vals_centro = []
                    for du in range(-5, 6):
                        for dv in range(-5, 6):
                            pu = int(np.clip(cx_d + du, 0, IMG_W - 1))
                            pv = int(np.clip(cy_d + dv, 0, IMG_H - 1))
                            d = depth_arr[pv, pu]
                            if 0.05 < d < 4.0:
                                vals_centro.append(d)
                    if not vals_centro:
                        time.sleep(0.01)
                        continue
                    z_box_cam = float(np.median(vals_centro))

                    # ── Arista superior: escaneo vertical buscando el quiebre ──
                    # (misma técnica que vision_brain_test.py)
                    z_arista  = 999.0
                    y_arista_d = y1_d
                    for fila in range(max(0, y1_d), min(IMG_H - 1, y2_d)):
                        fila_vals = []
                        for col in range(max(0, cx_d - 5), min(IMG_W - 1, cx_d + 6)):
                            dist = depth_arr[fila, col]
                            if 0.05 < dist < 4.0:
                                fila_vals.append(dist)
                        if fila_vals:
                            z_fila = float(np.median(fila_vals))
                            if z_fila < z_arista:
                                z_arista   = z_fila
                                y_arista_d = fila

                    if z_arista == 999.0:
                        z_arista = z_box_cam

                    # ── Proyección 3-D de la arista superior (extremos IZQ / DER) ──
                    x_cam_izq = ((x1_d - CX) * z_arista) / FOCAL_LENGTH
                    x_cam_der = ((x2_d - CX) * z_arista) / FOCAL_LENGTH
                    y_cam_arista = ((y_arista_d - CY) * z_arista) / FOCAL_LENGTH

                    # Centro horizontal de la caja (en profundidad del centro)
                    x_cam_centro = ((cx_d - CX) * z_box_cam) / FOCAL_LENGTH
                    y_cam_centro = ((cy_d - CY) * z_box_cam) / FOCAL_LENGTH

                    # Transformar todo al frame robot
                    center_base  = transform_camera_to_base(z_box_cam,  x_cam_centro, y_cam_centro)
                    edge_l_base  = transform_camera_to_base(z_arista,   x_cam_izq,    y_cam_arista)
                    edge_r_base  = transform_camera_to_base(z_arista,   x_cam_der,    y_cam_arista)

                    # ── Dimensiones reales estimadas ────────────────────────────
                    width_m  = abs(edge_l_base[1] - edge_r_base[1])  # eje Y robot
                    h_px     = y2_d - y1_d
                    height_m = (h_px * z_box_cam) / FOCAL_LENGTH

                    # ── Mesa ───────────────────────────────────────────────────
                    bbox_d  = (x1_d, y1_d, x2_d, y2_d)
                    z_mesa_cam, n_s = scan_table_around_box(depth_arr, bbox_d)
                    if z_mesa_cam is not None:
                        # Punto en la mesa: mismo X lateral que el centro, pero fila debajo de la caja
                        row_mesa = min(y2_d + 25, IMG_H - 1)
                        y_cam_mesa = ((row_mesa - CY) * z_mesa_cam) / FOCAL_LENGTH
                        mesa_base  = transform_camera_to_base(z_mesa_cam, x_cam_centro, y_cam_mesa)
                        z_mesa = mesa_base[2]
                    else:
                        z_mesa = center_base[2] - height_m / 2.0

                    detected = True
                    edge_l   = edge_l_base
                    edge_r   = edge_r_base

                    # ── HUD visual ─────────────────────────────────────────────
                    y_arista_c = int(y_arista_d / sy)
                    x1_vis, x2_vis = int(x1_c), int(x2_c)
                    cv2.line(annotated, (x1_vis, y_arista_c), (x2_vis, y_arista_c), (0, 255, 255), 2)
                    cv2.circle(annotated, (x1_vis, y_arista_c), 6, (255, 0, 255), -1)
                    cv2.circle(annotated, (x2_vis, y_arista_c), 6, (0, 165, 255), -1)
                    cv2.putText(annotated,
                                f"Centro robot: X{center_base[0]:.2f} Y{center_base[1]:.2f} Z{center_base[2]:.2f}",
                                (x1_vis, max(20, y_arista_c - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)

                with self.lock:
                    self.box_detected      = detected
                    self.box_center_base   = center_base
                    self.z_mesa_base       = z_mesa
                    self.box_width_cam     = width_m
                    self.box_height_cam    = height_m
                    self.edge_x_left_base  = edge_l
                    self.edge_x_right_base = edge_r
                    self.depth_frame_latest = depth_arr.copy()
                    self.annotated_frame   = annotated.copy()

            except Exception as e:
                print(f"[VISIÓN] Error en loop: {e}")

            elapsed = time.time() - t0
            time.sleep(max(0.0, 0.033 - elapsed))  # ~30 Hz

    def get_snapshot(self):
        """Devuelve una copia segura del último estado de visión."""
        with self.lock:
            return {
                'detected':    self.box_detected,
                'center':      self.box_center_base.copy() if self.box_center_base is not None else None,
                'z_mesa':      self.z_mesa_base,
                'width':       self.box_width_cam,
                'height':      self.box_height_cam,
                'edge_left':   self.edge_x_left_base.copy() if self.edge_x_left_base is not None else None,
                'edge_right':  self.edge_x_right_base.copy() if self.edge_x_right_base is not None else None,
                'depth_frame': self.depth_frame_latest,
                'frame_vis':   self.annotated_frame,
            }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MÓDULO IK BIMANUAL  (basado en vision_brain.py IntegratedIK +
#                          arquitectura de publicación de inverse_kinematics.py)
# ──────────────────────────────────────────────────────────────────────────────

class BimanalIK(Node):
    """
    Nodo ROS2 que:
      - Se suscribe a /lowstate para leer posición actual de articulaciones.
      - Publica en /arm_sdk (LowCmd) a 50 Hz.
      - Calcula IK para AMBOS brazos de forma independiente (no espejo).
    """

    def __init__(self):
        super().__init__('vision_brain_real')

        self.current_jpos  = [0.0] * 29
        self.state_received = False

        self.cmd_pub  = self.create_publisher(LowCmd, '/arm_sdk', 10)
        self.state_sub = self.create_subscription(
            LowState, '/lowstate', self._state_callback, 10)

        # ── Pinocchio ──────────────────────────────────────────────────────────
        try:
            full_model = pin.buildModelFromUrdf(URDF_PATH)
            # Bloqueamos piernas y cintura (igual que vision_brain.py)
            lock_names = [
                "left_hip_pitch_joint",   "left_hip_roll_joint",    "left_hip_yaw_joint",
                "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint",  "right_hip_roll_joint",   "right_hip_yaw_joint",
                "right_knee_joint",       "right_ankle_pitch_joint","right_ankle_roll_joint",
                "waist_yaw_joint",        "waist_roll_joint",       "waist_pitch_joint",
            ]
            locked_ids = [full_model.getJointId(j)
                          for j in lock_names if full_model.existJointName(j)]
            q_neutral  = pin.neutral(full_model)
            self.model = pin.buildReducedModel(full_model, locked_ids, q_neutral)
            self.data  = self.model.createData()

            self.left_hand_id  = self.model.getFrameId("left_rubber_hand")
            self.right_hand_id = self.model.getFrameId("right_rubber_hand")

            # Índice q del shoulder_roll para apertura inicial
            self.q_idx_l_roll = self.model.joints[
                self.model.getJointId("left_shoulder_roll_joint")].idx_q
            self.q_idx_r_roll = self.model.joints[
                self.model.getJointId("right_shoulder_roll_joint")].idx_q

        except Exception as e:
            self.get_logger().error(f"[IK] Error cargando URDF: {e}")
            self.model = None

        # Nombres y mapeo articular
        self.left_arm_names = [
            "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",    "left_elbow_joint",
            "left_wrist_roll_joint",      "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]
        self.right_arm_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",   "right_elbow_joint",
            "right_wrist_roll_joint",     "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        self.pin_to_g1_q     = {}
        self.left_v_indices  = []
        self.right_v_indices = []

        if self.model is not None:
            for i, name in enumerate(self.left_arm_names):
                if self.model.existJointName(name):
                    jid = self.model.getJointId(name)
                    self.pin_to_g1_q[self.model.joints[jid].idx_q] = G1_ARM_LEFT[i]
                    self.left_v_indices.append(self.model.joints[jid].idx_v)
            for i, name in enumerate(self.right_arm_names):
                if self.model.existJointName(name):
                    jid = self.model.getJointId(name)
                    self.pin_to_g1_q[self.model.joints[jid].idx_q] = G1_ARM_RIGHT[i]
                    self.right_v_indices.append(self.model.joints[jid].idx_v)

        self.q_math = pin.neutral(self.model) if self.model else None

        # Estado IK
        self.active_ik    = False
        self.target_l     = None
        self.target_r     = None
        self.traj_l       = []
        self.traj_r       = []
        self.final_tgt_l  = None
        self.final_tgt_r  = None

        self.hand_l_actual = np.zeros(3)
        self.hand_r_actual = np.zeros(3)

        self.lock_shoulder_roll  = False
        self.lock_elbows_wrists  = False
        self.use_6d              = False
        self.target_rot_l        = None
        self.target_rot_r        = None

        # Timer de control a 50 Hz
        self.timer = self.create_timer(DT, self._control_loop)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _state_callback(self, msg: LowState):
        for i in range(29):
            if i < len(msg.motor_state):
                self.current_jpos[i] = msg.motor_state[i].q
        if not self.state_received:
            self.state_received = True
            self.sync_with_reality()
            self.get_logger().info("[IK] Estado recibido. Robot listo.")

    def sync_with_reality(self):
        """Sincroniza el modelo Pinocchio con los encoders reales."""
        if self.model is None:
            return
        for q_idx, g1_idx in self.pin_to_g1_q.items():
            self.q_math[q_idx] = self.current_jpos[g1_idx]
        pin.forwardKinematics(self.model, self.data, self.q_math)
        pin.updateFramePlacements(self.model, self.data)
        self.hand_l_actual = self.data.oMf[self.left_hand_id].translation.copy()
        self.hand_r_actual = self.data.oMf[self.right_hand_id].translation.copy()

    # ── Trayectorias ──────────────────────────────────────────────────────────

    def _make_traj(self, start, end, step=0.02):
        dist = np.linalg.norm(end - start)
        n    = max(1, int(dist / step))
        return [start + (i / n) * (end - start) for i in range(1, n + 1)]

    def set_targets(self, tgt_l, tgt_r):
        pin.forwardKinematics(self.model, self.data, self.q_math)
        pin.updateFramePlacements(self.model, self.data)
        self.hand_l_actual = self.data.oMf[self.left_hand_id].translation.copy()
        self.hand_r_actual = self.data.oMf[self.right_hand_id].translation.copy()

        self.traj_l     = self._make_traj(self.hand_l_actual, tgt_l)
        self.traj_r     = self._make_traj(self.hand_r_actual, tgt_r)
        self.target_l   = self.traj_l.pop(0) if self.traj_l else tgt_l
        self.target_r   = self.traj_r.pop(0) if self.traj_r else tgt_r
        self.final_tgt_l = tgt_l.copy()
        self.final_tgt_r = tgt_r.copy()
        self.active_ik  = True

    def get_max_error(self):
        if self.final_tgt_l is None or self.final_tgt_r is None:
            return 999.0
        return max(np.linalg.norm(self.final_tgt_l - self.hand_l_actual),
                   np.linalg.norm(self.final_tgt_r - self.hand_r_actual))

    # ── Paso IK para un brazo ─────────────────────────────────────────────────

    def _ik_step(self, frame_id, v_indices, hand_actual, target, traj):
        err = target - hand_actual
        if np.linalg.norm(err) < 0.01 and traj:
            target = traj.pop(0)
            err    = target - hand_actual

        err_norm = np.linalg.norm(err)
        dq_arm   = np.zeros(len(v_indices))

        if self.use_6d and (self.target_rot_l is not None or self.target_rot_r is not None):
            # Modo 6D (posición + orientación)
            J     = pin.computeFrameJacobian(self.model, self.data, self.q_math,
                                             frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_arm = J[:, v_indices]
            if self.lock_elbows_wrists: J_arm[:, 3:] = 0.0
            if self.lock_shoulder_roll: J_arm[:, 1]  = 0.0

            target_rot = (self.target_rot_l if frame_id == self.left_hand_id
                          else self.target_rot_r)
            R_curr = self.data.oMf[frame_id].rotation
            R_err  = target_rot @ R_curr.T
            theta  = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
            w      = np.zeros(3)
            if theta > 1e-5:
                w = (theta / (2 * np.sin(theta))) * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1],
                ])
            err6 = np.concatenate([err, w])
            if np.linalg.norm(err6) > 0.04:
                err6 = err6 / np.linalg.norm(err6) * 0.04
            pi = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + (0.05**2) * np.eye(6))
            dq_arm = pi @ err6 * 3.0

        else:
            # Modo posición pura (3D)
            if err_norm > 0.03:
                err = err / err_norm * 0.03
            if np.linalg.norm(err) > 0.005:
                J     = pin.computeFrameJacobian(self.model, self.data, self.q_math,
                                                 frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_arm = J[:3, v_indices]
                if self.lock_elbows_wrists: J_arm[:, 3:] = 0.0
                if self.lock_shoulder_roll: J_arm[:, 1]  = 0.0
                pi = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + (0.05**2) * np.eye(3))
                dq_arm = pi @ err * 3.0

        return dq_arm, target

    # ── Loop de control a 50 Hz ───────────────────────────────────────────────

    def _control_loop(self):
        if not self.state_received or self.model is None:
            return

        cmd = LowCmd()

        if self.active_ik and self.target_l is not None and self.target_r is not None:
            pin.forwardKinematics(self.model, self.data, self.q_math)
            pin.updateFramePlacements(self.model, self.data)
            self.hand_l_actual = self.data.oMf[self.left_hand_id].translation.copy()
            self.hand_r_actual = self.data.oMf[self.right_hand_id].translation.copy()

            dq = np.zeros(self.model.nv)

            dq_l, self.target_l = self._ik_step(
                self.left_hand_id,  self.left_v_indices,
                self.hand_l_actual, self.target_l, self.traj_l)
            for i, vi in enumerate(self.left_v_indices):
                dq[vi] = dq_l[i]

            dq_r, self.target_r = self._ik_step(
                self.right_hand_id, self.right_v_indices,
                self.hand_r_actual, self.target_r, self.traj_r)
            for i, vi in enumerate(self.right_v_indices):
                dq[vi] = dq_r[i]

            self.q_math = pin.integrate(self.model, self.q_math, dq * DT)

        # Enviar comandos a todos los motores de brazos
        for q_idx, g1_idx in self.pin_to_g1_q.items():
            cmd.motor_cmd[g1_idx].q   = float(self.q_math[q_idx])
            cmd.motor_cmd[g1_idx].dq  = 0.0
            cmd.motor_cmd[g1_idx].tau = 0.0
            cmd.motor_cmd[g1_idx].kp  = KP
            cmd.motor_cmd[g1_idx].kd  = KD

        # Cintura bloqueada en posición actual (igual que inverse_kinematics.py)
        for wi in G1_WAIST:
            cmd.motor_cmd[wi].q  = self.current_jpos[wi]
            cmd.motor_cmd[wi].kp = KP * 4.0
            cmd.motor_cmd[wi].kd = KD * 4.0

        # Token de habilitación /arm_sdk (obligatorio)
        cmd.motor_cmd[NOT_USED_JOINT].q = 1.0
        self.cmd_pub.publish(cmd)

    def release(self):
        """Devuelve el control al robot (desactiva /arm_sdk)."""
        cmd = LowCmd()
        cmd.motor_cmd[NOT_USED_JOINT].q = 0.0
        self.cmd_pub.publish(cmd)
        time.sleep(0.3)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  MÁQUINA DE ESTADOS  (main loop)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()

    # Lanzar hilo de visión antes de entrar al spin de ROS2
    visión = VisionModule()
    visión.start()

    robot = BimanalIK()

    # Girar el nodo en un hilo separado para que el loop principal funcione
    spin_thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    spin_thread.start()

    # Esperar a que llegue el primer estado del robot
    print("[MAIN] Esperando estado del robot...")
    while not robot.state_received:
        time.sleep(0.1)
    print("[MAIN] Estado recibido. Iniciando máquina de estados.")

    # ── Variables de estado ────────────────────────────────────────────────────
    estado           = "ANALIZANDO_ESCENA"
    tiempo_estado    = time.time()
    memoria_caja     = {}
    LIMITE_Z_SEGURO  = -999.0

    # Para el descenso controlado
    z_descenso_actual  = 0.0
    z_descenso_obj     = 0.0
    VELOCIDAD_DESCENSO = 0.006   # m por tick × 50 Hz ≈ 0.3 m/s

    nombre_ventana = "Vision Brain Real G1"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            t_loop = time.time()
            now    = t_loop

            # ── Snapshot de visión ──────────────────────────────────────────────
            snap = visión.get_snapshot()
            frame_vis = snap['frame_vis']
            if frame_vis is None:
                frame_vis = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

            # ── Proyección de manos sobre imagen (HUD) ─────────────────────────
            if robot.state_received and LIMITE_Z_SEGURO != -999.0:
                for mano, color in [(robot.hand_l_actual, (255, 100, 0)),
                                    (robot.hand_r_actual, (0, 100, 255))]:
                    u_m, v_m = project_base_to_camera(mano)
                    if u_m is not None:
                        cv2.circle(frame_vis, (u_m, v_m), 7, color, -1)

            cv2.putText(frame_vis, f"ESTADO: {estado}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ══════════════════════════════════════════════════════════════════
            #  ANALIZANDO_ESCENA: observar y guardar geometría de la caja
            # ══════════════════════════════════════════════════════════════════
            if estado == "ANALIZANDO_ESCENA":
                if not snap['detected']:
                    cv2.putText(frame_vis, "Buscando caja...",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    centro = snap['center']
                    z_mesa = snap['z_mesa']
                    if centro is None or z_mesa is None:
                        pass
                    else:
                        LIMITE_Z_SEGURO = z_mesa + 0.05   # 5 cm por encima de la mesa
                        memoria_caja = {
                            'centro': centro.copy(),
                            'z_mesa': z_mesa,
                            'width':  max(snap['width'], ANCHO_CAJA_REAL),   # usar dato real si es fiable
                            'height': snap['height'],
                            'edge_l': snap['edge_left'],
                            'edge_r': snap['edge_right'],
                        }
                        print(f"[ESCENA] Centro caja (robot): {centro}")
                        print(f"[ESCENA] Z mesa: {z_mesa:.3f}  LIMITE_Z: {LIMITE_Z_SEGURO:.3f}")
                        print(f"[ESCENA] Ancho estimado: {memoria_caja['width']:.3f} m")

                        robot.active_ik = False
                        robot.sync_with_reality()

                        estado        = "ABRIR_BRAZOS"
                        tiempo_estado = now

            # ══════════════════════════════════════════════════════════════════
            #  ABRIR_BRAZOS: apertura lateral de hombros (igual que vision_brain)
            # ══════════════════════════════════════════════════════════════════
            elif estado == "ABRIR_BRAZOS":
                if not hasattr(robot, '_base_roll_l'):
                    robot._base_roll_l = robot.q_math[robot.q_idx_l_roll]
                    robot._base_roll_r = robot.q_math[robot.q_idx_r_roll]

                progreso = min(1.0, (now - tiempo_estado) / 1.2)
                robot.q_math[robot.q_idx_l_roll] = robot._base_roll_l + 0.6 * progreso
                robot.q_math[robot.q_idx_r_roll] = robot._base_roll_r - 0.6 * progreso

                if progreso >= 1.0:
                    print("[ESTADO] Brazos abiertos. Pasando a CALCULAR_PREPARAR.")
                    estado        = "CALCULAR_PREPARAR"
                    tiempo_estado = now

            # ══════════════════════════════════════════════════════════════════
            #  CALCULAR_PREPARAR: lleva los brazos a la altura y Y correctas,
            #  ligeramente por detrás del borde frontal de la caja
            # ══════════════════════════════════════════════════════════════════
            elif estado == "CALCULAR_PREPARAR":
                centro      = memoria_caja['centro']
                medio_ancho = memoria_caja['width'] / 2.0
                target_z    = max(memoria_caja['z_mesa'] + 0.15, LIMITE_Z_SEGURO + 0.05)

                tgt_l = np.array([
                    centro[0] - 0.10,                  # ligeramente por detrás en X
                    centro[1] + medio_ancho + 0.15,    # a la izquierda de la caja
                    target_z,
                ])
                tgt_r = np.array([
                    centro[0] - 0.10,
                    centro[1] - medio_ancho - 0.15,    # a la derecha de la caja
                    target_z,
                ])

                robot.lock_shoulder_roll  = True
                robot.lock_elbows_wrists  = False
                robot.use_6d              = False
                robot.set_targets(tgt_l, tgt_r)
                print(f"[IK] Preparar → L:{tgt_l}  R:{tgt_r}")
                estado        = "MOVIENDO_PREPARAR"
                tiempo_estado = now

            elif estado == "MOVIENDO_PREPARAR":
                if robot.get_max_error() < 0.04 or (now - tiempo_estado > 5.0):
                    print("[ESTADO] Posición preparada. Pasando a POSICION_SOBRE.")
                    estado        = "POSICION_SOBRE"
                    tiempo_estado = now

            # ══════════════════════════════════════════════════════════════════
            #  POSICION_SOBRE: manos directamente sobre los puntos de agarre,
            #  a Z alta (por encima de la caja)
            # ══════════════════════════════════════════════════════════════════
            elif estado == "POSICION_SOBRE":
                centro      = memoria_caja['centro']
                medio_ancho = memoria_caja['width'] / 2.0
                z_sobre     = max(LIMITE_Z_SEGURO + 0.18, centro[2] + 0.12)

                # X apunta al centro de la cara visible de la caja
                # (frente de la caja + mitad de profundidad - longitud del gripper)
                tgt_x = centro[0] + (PROFUNDIDAD_CAJA_REAL / 2.0) - LONGITUD_MANO

                tgt_l = np.array([tgt_x, centro[1] + medio_ancho + 0.15, z_sobre])
                tgt_r = np.array([tgt_x, centro[1] - medio_ancho - 0.15, z_sobre])

                robot.lock_shoulder_roll = True
                robot.use_6d             = False
                robot.set_targets(tgt_l, tgt_r)
                print(f"[IK] Sobre la caja → L:{tgt_l}  R:{tgt_r}")
                estado        = "MOVIENDO_SOBRE"
                tiempo_estado = now

            elif estado == "MOVIENDO_SOBRE":
                if robot.get_max_error() < 0.04 or (now - tiempo_estado > 5.0):
                    print("[ESTADO] Manos sobre la caja. Pasando a ALINEAR_PALMAS.")
                    estado        = "ALINEAR_PALMAS"
                    tiempo_estado = now

            # ══════════════════════════════════════════════════════════════════
            #  ALINEAR_PALMAS: rota palmas hacia dentro (modo 6D)
            # ══════════════════════════════════════════════════════════════════
            elif estado == "ALINEAR_PALMAS":
                pin.forwardKinematics(robot.model, robot.data, robot.q_math)
                pin.updateFramePlacements(robot.model, robot.data)

                ang = 0.8   # ~45°
                # Brazo izquierdo: roll negativo para que la palma mire a la derecha
                R_l = np.array([
                    [1, 0,           0           ],
                    [0, math.cos(-ang), -math.sin(-ang)],
                    [0, math.sin(-ang),  math.cos(-ang)],
                ])
                # Brazo derecho: roll positivo (simétrico)
                R_r = np.array([
                    [1, 0,          0          ],
                    [0, math.cos(ang), -math.sin(ang)],
                    [0, math.sin(ang),  math.cos(ang)],
                ])
                robot.target_rot_l = R_l @ robot.data.oMf[robot.left_hand_id].rotation.copy()
                robot.target_rot_r = R_r @ robot.data.oMf[robot.right_hand_id].rotation.copy()
                robot.use_6d       = True
                robot.lock_shoulder_roll = True

                centro      = memoria_caja['centro']
                medio_ancho = memoria_caja['width'] / 2.0
                z_sobre     = max(LIMITE_Z_SEGURO + 0.18, centro[2] + 0.12)
                tgt_x       = centro[0] + (PROFUNDIDAD_CAJA_REAL / 2.0) - LONGITUD_MANO

                tgt_l = np.array([tgt_x, centro[1] + medio_ancho + 0.15, z_sobre])
                tgt_r = np.array([tgt_x, centro[1] - medio_ancho - 0.15, z_sobre])

                robot.set_targets(tgt_l, tgt_r)
                print("[ESTADO] Alineando palmas (6D). Pasando a MOVIENDO_ALINEAR.")
                estado        = "MOVIENDO_ALINEAR"
                tiempo_estado = now

            elif estado == "MOVIENDO_ALINEAR":
                if robot.get_max_error() < 0.05 or (now - tiempo_estado > 4.0):
                    # Preparar descenso
                    z_descenso_actual = float(robot.hand_l_actual[2])
                    z_mitad_caja      = (memoria_caja['centro'][2] + memoria_caja['z_mesa']) / 2.0
                    z_descenso_obj    = max(z_mitad_caja, LIMITE_Z_SEGURO)
                    print(f"[ESTADO] Inicio descenso: {z_descenso_actual:.3f} → {z_descenso_obj:.3f}")
                    estado        = "BAJAR_MANOS"
                    tiempo_estado = now

            # ══════════════════════════════════════════════════════════════════
            #  BAJAR_MANOS: descenso lento a la altura de agarre
            # ══════════════════════════════════════════════════════════════════
            elif estado == "BAJAR_MANOS":
                robot.lock_shoulder_roll = False

                puede_bajar = (z_descenso_actual - VELOCIDAD_DESCENSO) >= z_descenso_obj
                if puede_bajar:
                    z_descenso_actual -= VELOCIDAD_DESCENSO
                else:
                    z_descenso_actual = z_descenso_obj

                centro      = memoria_caja['centro']
                medio_ancho = memoria_caja['width'] / 2.0
                tgt_x       = centro[0] + (PROFUNDIDAD_CAJA_REAL / 2.0) - LONGITUD_MANO

                if not robot.traj_l and not robot.traj_r:
                    tgt_l = np.array([tgt_x, centro[1] + medio_ancho + 0.15, z_descenso_actual])
                    tgt_r = np.array([tgt_x, centro[1] - medio_ancho - 0.15, z_descenso_actual])
                    robot.set_targets(tgt_l, tgt_r)

                ya_llegamos = z_descenso_actual <= z_descenso_obj + 0.005
                timeout     = now - tiempo_estado > 7.0
                if ya_llegamos or timeout:
                    print(f"[ESTADO] Descenso completado. Pasando a CERRAR_AGARRE.")
                    estado        = "CERRAR_AGARRE"
                    tiempo_estado = now

            # ══════════════════════════════════════════════════════════════════
            #  CERRAR_AGARRE: mueve manos hacia el interior de la caja en Y
            # ══════════════════════════════════════════════════════════════════
            elif estado == "CERRAR_AGARRE":
                robot.lock_shoulder_roll = False

                centro      = memoria_caja['centro']
                medio_ancho = memoria_caja['width'] / 2.0
                PENETRACION = 0.045  # 4.5 cm dentro de los bordes de la caja

                borde_l = centro[1] + medio_ancho - PENETRACION
                borde_r = centro[1] - medio_ancho + PENETRACION

                dist_l = float(robot.hand_l_actual[1]) - borde_l
                dist_r = borde_r - float(robot.hand_r_actual[1])

                if (dist_l < 0.01 and dist_r < 0.01) or (now - tiempo_estado > 16.0):
                    print("[ESTADO] Agarre completado. Pasando a LEVANTAR_RETRAER.")
                    estado        = "LEVANTAR_RETRAER"
                    tiempo_estado = now

                elif not robot.traj_l and not robot.traj_r:
                    paso_l = min(0.01, max(0.0, dist_l))
                    paso_r = min(0.01, max(0.0, dist_r))
                    tgt_x  = centro[0] + (PROFUNDIDAD_CAJA_REAL / 2.0) - LONGITUD_MANO
                    z_agr  = z_descenso_obj

                    tgt_l = np.array([tgt_x, robot.hand_l_actual[1] - paso_l, z_agr])
                    tgt_r = np.array([tgt_x, robot.hand_r_actual[1] + paso_r, z_agr])
                    robot.set_targets(tgt_l, tgt_r)

            # ══════════════════════════════════════════════════════════════════
            #  LEVANTAR_RETRAER: fase 1 – tirar caja hacia el cuerpo
            # ══════════════════════════════════════════════════════════════════
            elif estado == "LEVANTAR_RETRAER":
                robot.lock_shoulder_roll = True
                tgt_l = robot.hand_l_actual.copy()
                tgt_r = robot.hand_r_actual.copy()
                tgt_l[0] = 0.20;  tgt_r[0] = 0.20    # X = 20 cm (cerca del cuerpo)
                tgt_l[2] += 0.05; tgt_r[2] += 0.05   # Pequeña subida para no arrastrar
                robot.set_targets(tgt_l, tgt_r)
                print("[ESTADO] Retrayendo al pecho. Pasando a MOVIENDO_RETRAER.")
                estado        = "MOVIENDO_RETRAER"
                tiempo_estado = now

            elif estado == "MOVIENDO_RETRAER":
                if robot.get_max_error() < 0.04 or (now - tiempo_estado > 4.0):
                    print("[ESTADO] Retracción completada. Pasando a LEVANTAR_SUBIR.")
                    estado        = "LEVANTAR_SUBIR"
                    tiempo_estado = now

            # ══════════════════════════════════════════════════════════════════
            #  LEVANTAR_SUBIR: fase 2 – subir la caja en vertical
            # ══════════════════════════════════════════════════════════════════
            elif estado == "LEVANTAR_SUBIR":
                tgt_l = robot.hand_l_actual.copy()
                tgt_r = robot.hand_r_actual.copy()
                tgt_l[2] = 0.40;  tgt_r[2] = 0.40   # Subir a 40 cm de altura
                robot.set_targets(tgt_l, tgt_r)
                print("[ESTADO] Subiendo caja. Pasando a MOVIENDO_SUBIR.")
                estado        = "MOVIENDO_SUBIR"
                tiempo_estado = now

            elif estado == "MOVIENDO_SUBIR":
                if robot.get_max_error() < 0.04 or (now - tiempo_estado > 5.0):
                    print("[ESTADO] ✅ Caja levantada con éxito. Estado: FINALIZADO.")
                    estado        = "FINALIZADO"
                    tiempo_estado = now

            elif estado == "FINALIZADO":
                cv2.putText(frame_vis, "¡CAJA LEVANTADA!",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            # ── Mostrar frame ──────────────────────────────────────────────────
            cv2.imshow(nombre_ventana, frame_vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(max(0.0, DT - (time.time() - t_loop)))

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupción de teclado. Liberando control.")

    finally:
        robot.release()
        visión.stop()
        cv2.destroyAllWindows()
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
