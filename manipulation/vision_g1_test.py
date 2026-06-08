"""
vision_brain_real.py
====================
Brain para el robot real Unitree G1 (29 DOF). Aislando DDS con Multiprocessing.
"""

# 1. DAR PRIORIDAD A PYTORCH/YOLO (Evita el error del bloque estático TLS)
from ultralytics import YOLO
import torch

# 2. Resto de librerías
import os, sys, time, threading, math
import multiprocessing
import numpy as np
import cv2
import zmq
import pyrealsense2 as rs
import pinocchio as pin

import rclpy
from rclpy.node import Node
from unitree_hg.msg import LowCmd, LowState

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PARÁMETROS GLOBALES
# ... (el resto del código sigue igual)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  PARÁMETROS GLOBALES
# ──────────────────────────────────────────────────────────────────────────────

CX             = 320.0
CY             = 240.0
FOCAL_LENGTH   = 460.0
IMG_W, IMG_H   = 640, 480

CAMERA_TILT_DEG   = 42.4
CAMERA_TILT_RAD   = math.radians(CAMERA_TILT_DEG)
CAMERA_OFFSET_X   =  0.04764
CAMERA_OFFSET_Y   = -0.020
CAMERA_OFFSET_Z   =  0.46268

ANCHO_CAJA_REAL       = 0.20
PROFUNDIDAD_CAJA_REAL = 0.20
LONGITUD_MANO         = 0.12

YOLO_MODEL_PATH = "best.pt"
YOLO_CONF       = 0.25

G1_ARM_LEFT    = [15, 16, 17, 18, 19, 20, 21]
G1_ARM_RIGHT   = [22, 23, 24, 25, 26, 27, 28]
G1_WAIST       = [12, 13, 14]
NOT_USED_JOINT = 29

KP = 60.0
KD = 1.5
DT = 0.02   

URDF_PATH = os.path.expanduser("~/robot_ws/src/g1pilot/description_files/urdf/g1_29dof.urdf")

# ──────────────────────────────────────────────────────────────────────────────
# 2.  FUNCIONES MATEMÁTICAS GLOBALES
# ──────────────────────────────────────────────────────────────────────────────

def transform_camera_to_base(z_dist, x_lateral, y_vertical):
    st = math.sin(CAMERA_TILT_RAD)
    ct = math.cos(CAMERA_TILT_RAD)
    x_robot = CAMERA_OFFSET_X + st * z_dist + ct * y_vertical
    y_robot = CAMERA_OFFSET_Y - x_lateral
    z_robot = CAMERA_OFFSET_Z - ct * z_dist + st * y_vertical
    return np.array([x_robot, y_robot, z_robot])

def project_base_to_camera(pos_base):
    st = math.sin(CAMERA_TILT_RAD)
    ct = math.cos(CAMERA_TILT_RAD)
    dx = pos_base[0] - CAMERA_OFFSET_X
    dy = pos_base[1] - CAMERA_OFFSET_Y
    dz = pos_base[2] - CAMERA_OFFSET_Z
    z_c = st * dx - ct * dz
    x_l = -dy
    y_v = ct * dx + st * dz

    if z_c < 0.05:
        return None, None
    u = int(np.clip(CX + (x_l * FOCAL_LENGTH) / z_c, 0, IMG_W - 1))
    v = int(np.clip(CY + (y_v * FOCAL_LENGTH) / z_c, 0, IMG_H - 1))
    return u, v

def scan_table_around_box(depth_frame, bbox):
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
    for row_off in range(5, 90, 3):
        row = int(np.clip(y2_d + row_off, 0, IMG_H - 1))
        for col_off in range(-w_d // 2, int(w_d * 1.5), 4):
            col = int(np.clip(x1_d + col_off, 0, IMG_W - 1))
            d = depth_frame[row, col]
            if 0.05 < d < box_z_cam * 1.6: samples.append(d)
    for col_off in range(w_d // 2 + 5, w_d + 100, 4):
        col = int(np.clip(x1_d - col_off, 0, IMG_W - 1))
        for row_off in range(-h_d // 3, h_d + 30, 4):
            row = int(np.clip(y1_d + row_off, 0, IMG_H - 1))
            d = depth_frame[row, col]
            if 0.05 < d < box_z_cam * 1.6: samples.append(d)
    for col_off in range(w_d // 2 + 5, w_d + 100, 4):
        col = int(np.clip(x2_d + col_off, 0, IMG_W - 1))
        for row_off in range(-h_d // 3, h_d + 30, 4):
            row = int(np.clip(y1_d + row_off, 0, IMG_H - 1))
            d = depth_frame[row, col]
            if 0.05 < d < box_z_cam * 1.6: samples.append(d)
    if len(samples) < 15:
        return box_z_cam, len(samples)
    return float(np.percentile(samples, 15)), len(samples)

# ──────────────────────────────────────────────────────────────────────────────
# 3.  PROCESO DE VISIÓN AISLADO (No choca con ROS2)
# ──────────────────────────────────────────────────────────────────────────────

class VisionProcess(multiprocessing.Process):
    def __init__(self, shared_data):
        super().__init__()
        self.shared_data = shared_data
        self.daemon = True

    def run(self):
        # ⚠️ IMPORTACIÓN LOCAL: Aísla el SDK de Unitree estrictamente a este proceso
        sdk_path = "/root/unitree_sdk2_python"
        if sdk_path not in sys.path:
            sys.path.append(sdk_path)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.video.video_client import VideoClient

        try:
            ChannelFactoryInitialize(0, "eth0")
            print("[VISIÓN] Unitree SDK inicializado en eth0.")
        except Exception as e:
            print(f"[VISIÓN] Error Unitree SDK: {e}")

        video = VideoClient()
        video.Init()

        context_zmq = zmq.Context()
        zmq_pub = context_zmq.socket(zmq.PUB)
        zmq_pub.bind("tcp://0.0.0.0:6001")
        print("[VISIÓN] Transmisión ZMQ activa en puerto 6001.")

        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
        pipeline.start(cfg)
        print("[VISIÓN] RealSense depth conectado.")

        model = YOLO(YOLO_MODEL_PATH)
        print("[VISIÓN] Modelo YOLO cargado.")

        while True:
            t0 = time.time()
            try:
                ret, data = video.GetImageSample()
                frames = pipeline.wait_for_frames()
                depth_frame_rs = frames.get_depth_frame()

                if ret != 0 or not data or not depth_frame_rs:
                    time.sleep(0.01)
                    continue

                np_arr = np.frombuffer(bytes(data), np.uint8)
                color_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if color_img is None: continue

                depth_arr = np.asanyarray(depth_frame_rs.get_data()).astype(np.float32) * depth_frame_rs.get_units()

                h_c, w_c = color_img.shape[:2]
                sx = IMG_W / w_c
                sy = IMG_H / h_c

                results = model.predict(color_img, conf=YOLO_CONF, verbose=False)
                
                detected = False
                if len(results[0].boxes) > 0:
                    annotated = results[0].plot()
        cajas_validas = []
    
        for box in results[0].boxes:
        # 1. Filtro de Clase: Asegúrate de que el ID (ej: 0) corresponde a tu caja
        # Cambia el 0 por el ID de tu clase si es diferente.
            if int(box.cls[0]) != 0: 
                continue
            
            x1_c, y1_c, x2_c, y2_c = map(int, box.xyxy[0])
        
        # 2. Filtro de Posición (Censurar los brazos):
        # Ignorar detecciones que estén en el tercio inferior de la imagen
        # (que es por donde asoman los brazos del G1)
            if y2_c > IMG_H * 0.85: 
                continue
            
            cajas_validas.append(box)

        if cajas_validas:
        # Ahora sí, de las válidas, cogemos la de mayor confianza
            best_box = max(cajas_validas, key=lambda b: float(b.conf[0]))
            x1_c, y1_c, x2_c, y2_c = map(int, best_box.xyxy[0])
        else:
        # Si todo lo que vio eran brazos, ignoramos este frame
            detected = False

                    x1_d = int(np.clip(x1_c * sx, 0, IMG_W - 1))
                    y1_d = int(np.clip(y1_c * sy, 0, IMG_H - 1))
                    x2_d = int(np.clip(x2_c * sx, 0, IMG_W - 1))
                    y2_d = int(np.clip(y2_c * sy, 0, IMG_H - 1))

                    cx_d = (x1_d + x2_d) // 2
                    cy_d = (y1_d + y2_d) // 2

                    vals_centro = [depth_arr[int(np.clip(cy_d+dv, 0, IMG_H-1)), int(np.clip(cx_d+du, 0, IMG_W-1))] 
                                   for du in range(-5,6) for dv in range(-5,6) 
                                   if 0.05 < depth_arr[int(np.clip(cy_d+dv, 0, IMG_H-1)), int(np.clip(cx_d+du, 0, IMG_W-1))] < 4.0]
                    
                    if vals_centro:
                        z_box_cam = float(np.median(vals_centro))
                        z_arista  = 999.0
                        y_arista_d = y1_d

                        for fila in range(max(0, y1_d), min(IMG_H - 1, y2_d)):
                            fila_vals = [depth_arr[fila, col] for col in range(max(0, cx_d - 5), min(IMG_W - 1, cx_d + 6)) if 0.05 < depth_arr[fila, col] < 4.0]
                            if fila_vals:
                                z_fila = float(np.median(fila_vals))
                                if z_fila < z_arista:
                                    z_arista, y_arista_d = z_fila, fila

                        if z_arista == 999.0: z_arista = z_box_cam

                        x_cam_izq = ((x1_d - CX) * z_arista) / FOCAL_LENGTH
                        x_cam_der = ((x2_d - CX) * z_arista) / FOCAL_LENGTH
                        y_cam_arista = ((y_arista_d - CY) * z_arista) / FOCAL_LENGTH
                        x_cam_centro = ((cx_d - CX) * z_box_cam) / FOCAL_LENGTH
                        y_cam_centro = ((cy_d - CY) * z_box_cam) / FOCAL_LENGTH

                        center_base  = transform_camera_to_base(z_box_cam,  x_cam_centro, y_cam_centro)
                        edge_l_base  = transform_camera_to_base(z_arista,   x_cam_izq,    y_cam_arista)
                        edge_r_base  = transform_camera_to_base(z_arista,   x_cam_der,    y_cam_arista)

                        width_m  = abs(edge_l_base[1] - edge_r_base[1])
                        height_m = ((y2_d - y1_d) * z_box_cam) / FOCAL_LENGTH

                        z_mesa_cam, _ = scan_table_around_box(depth_arr, (x1_d, y1_d, x2_d, y2_d))
                        if z_mesa_cam is not None:
                            y_cam_mesa = ((min(y2_d + 25, IMG_H - 1) - CY) * z_mesa_cam) / FOCAL_LENGTH
                            z_mesa = transform_camera_to_base(z_mesa_cam, x_cam_centro, y_cam_mesa)[2]
                        else:
                            z_mesa = center_base[2] - height_m / 2.0

                        detected = True
                        y_arista_c, x1_vis, x2_vis = int(y_arista_d / sy), int(x1_c), int(x2_c)
                        cv2.line(annotated, (x1_vis, y_arista_c), (x2_vis, y_arista_c), (0, 255, 255), 2)
                        cv2.circle(annotated, (x1_vis, y_arista_c), 6, (255, 0, 255), -1)
                        cv2.circle(annotated, (x2_vis, y_arista_c), 6, (0, 165, 255), -1)

                # ── Sincronización segura de datos ──
                self.shared_data['detected'] = detected
                if detected:
                    self.shared_data['center'] = center_base
                    self.shared_data['z_mesa'] = z_mesa
                    self.shared_data['width']  = width_m
                    self.shared_data['height'] = height_m
                    self.shared_data['edge_l'] = edge_l_base
                    self.shared_data['edge_r'] = edge_r_base

                # ── Construir y enviar HUD ZMQ ──
                frame_to_send = annotated if detected else color_img
                estado_act = self.shared_data.get('estado', 'INICIANDO...')
                cv2.putText(frame_to_send, f"ESTADO: {estado_act}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if self.shared_data.get('limite_z', -999.0) != -999.0:
                    mano_l = self.shared_data.get('mano_l', None)
                    mano_r = self.shared_data.get('mano_r', None)
                    if mano_l is not None:
                        u_m, v_m = project_base_to_camera(mano_l)
                        if u_m is not None: cv2.circle(frame_to_send, (u_m, v_m), 7, (255, 100, 0), -1)
                    if mano_r is not None:
                        u_m, v_m = project_base_to_camera(mano_r)
                        if u_m is not None: cv2.circle(frame_to_send, (u_m, v_m), 7, (0, 100, 255), -1)

                _, buffer = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 80])
                zmq_pub.send(buffer.tobytes())

            except Exception as e:
                pass
            
            elapsed = time.time() - t0
            time.sleep(max(0.0, 0.033 - elapsed))

# ──────────────────────────────────────────────────────────────────────────────
# 4.  MÓDULO IK BIMANUAL ROS2
# ──────────────────────────────────────────────────────────────────────────────

class BimanalIK(Node):
    def __init__(self):
        super().__init__('vision_brain_real')
        self.current_jpos  = [0.0] * 29
        self.state_received = False
        self.cmd_pub  = self.create_publisher(LowCmd, '/arm_sdk', 10)
        self.state_sub = self.create_subscription(LowState, '/lowstate', self._state_callback, 10)

        try:
            full_model = pin.buildModelFromUrdf(URDF_PATH)
            lock_names = [
                "left_hip_pitch_joint",   "left_hip_roll_joint",    "left_hip_yaw_joint",
                "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint",  "right_hip_roll_joint",   "right_hip_yaw_joint",
                "right_knee_joint",       "right_ankle_pitch_joint","right_ankle_roll_joint",
                "waist_yaw_joint",        "waist_roll_joint",       "waist_pitch_joint",
            ]
            locked_ids = [full_model.getJointId(j) for j in lock_names if full_model.existJointName(j)]
            q_neutral  = pin.neutral(full_model)
            self.model = pin.buildReducedModel(full_model, locked_ids, q_neutral)
            self.data  = self.model.createData()

            self.left_hand_id  = self.model.getFrameId("left_rubber_hand")
            self.right_hand_id = self.model.getFrameId("right_rubber_hand")
            self.q_idx_l_roll = self.model.joints[self.model.getJointId("left_shoulder_roll_joint")].idx_q
            self.q_idx_r_roll = self.model.joints[self.model.getJointId("right_shoulder_roll_joint")].idx_q

        except Exception as e:
            self.get_logger().error(f"[IK] Error cargando URDF: {e}")
            self.model = None

        self.left_arm_names = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
        self.right_arm_names = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]

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
        self.active_ik, self.target_l, self.target_r = False, None, None
        self.traj_l, self.traj_r = [], []
        self.final_tgt_l, self.final_tgt_r = None, None
        self.hand_l_actual, self.hand_r_actual = np.zeros(3), np.zeros(3)
        self.lock_shoulder_roll, self.lock_elbows_wrists, self.use_6d = False, False, False
        self.target_rot_l, self.target_rot_r = None, None

        self.timer = self.create_timer(DT, self._control_loop)

    def _state_callback(self, msg: LowState):
        for i in range(29):
            if i < len(msg.motor_state): self.current_jpos[i] = msg.motor_state[i].q
        if not self.state_received:
            self.state_received = True
            self.sync_with_reality()
            self.get_logger().info("[IK] Estado recibido. Robot listo.")

    def sync_with_reality(self):
        if self.model is None: return
        for q_idx, g1_idx in self.pin_to_g1_q.items(): self.q_math[q_idx] = self.current_jpos[g1_idx]
        pin.forwardKinematics(self.model, self.data, self.q_math)
        pin.updateFramePlacements(self.model, self.data)
        self.hand_l_actual = self.data.oMf[self.left_hand_id].translation.copy()
        self.hand_r_actual = self.data.oMf[self.right_hand_id].translation.copy()

    def _make_traj(self, start, end, step=0.02):
        dist = np.linalg.norm(end - start)
        n = max(1, int(dist / step))
        return [start + (i / n) * (end - start) for i in range(1, n + 1)]

    def set_targets(self, tgt_l, tgt_r):
        self.sync_with_reality()
        self.traj_l = self._make_traj(self.hand_l_actual, tgt_l)
        self.traj_r = self._make_traj(self.hand_r_actual, tgt_r)
        self.target_l = self.traj_l.pop(0) if self.traj_l else tgt_l
        self.target_r = self.traj_r.pop(0) if self.traj_r else tgt_r
        self.final_tgt_l, self.final_tgt_r = tgt_l.copy(), tgt_r.copy()
        self.active_ik = True

    def get_max_error(self):
        if self.final_tgt_l is None or self.final_tgt_r is None: return 999.0
        return max(np.linalg.norm(self.final_tgt_l - self.hand_l_actual), np.linalg.norm(self.final_tgt_r - self.hand_r_actual))

    def _ik_step(self, frame_id, v_indices, hand_actual, target, traj):
        err = target - hand_actual
        if np.linalg.norm(err) < 0.01 and traj:
            target = traj.pop(0)
            err = target - hand_actual
        err_norm = np.linalg.norm(err)
        dq_arm = np.zeros(len(v_indices))

        if self.use_6d and (self.target_rot_l is not None or self.target_rot_r is not None):
            J = pin.computeFrameJacobian(self.model, self.data, self.q_math, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_arm = J[:, v_indices]
            if self.lock_elbows_wrists: J_arm[:, 3:] = 0.0
            if self.lock_shoulder_roll: J_arm[:, 1]  = 0.0

            target_rot = (self.target_rot_l if frame_id == self.left_hand_id else self.target_rot_r)
            R_err = target_rot @ self.data.oMf[frame_id].rotation.T
            theta = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
            w = np.zeros(3)
            if theta > 1e-5:
                w = (theta / (2 * np.sin(theta))) * np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
            err6 = np.concatenate([err, w])
            if np.linalg.norm(err6) > 0.04: err6 = err6 / np.linalg.norm(err6) * 0.04
            pi = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + (0.05**2) * np.eye(6))
            dq_arm = pi @ err6 * 3.0
        else:
            if err_norm > 0.03: err = err / err_norm * 0.03
            if np.linalg.norm(err) > 0.005:
                J = pin.computeFrameJacobian(self.model, self.data, self.q_math, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                J_arm = J[:3, v_indices]
                if self.lock_elbows_wrists: J_arm[:, 3:] = 0.0
                if self.lock_shoulder_roll: J_arm[:, 1]  = 0.0
                pi = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + (0.05**2) * np.eye(3))
                dq_arm = pi @ err * 3.0

        return dq_arm, target

    def _control_loop(self):
        if not self.state_received or self.model is None: return
        cmd = LowCmd()

        if self.active_ik and self.target_l is not None and self.target_r is not None:
            pin.forwardKinematics(self.model, self.data, self.q_math)
            pin.updateFramePlacements(self.model, self.data)
            self.hand_l_actual = self.data.oMf[self.left_hand_id].translation.copy()
            self.hand_r_actual = self.data.oMf[self.right_hand_id].translation.copy()

            dq = np.zeros(self.model.nv)
            dq_l, self.target_l = self._ik_step(self.left_hand_id, self.left_v_indices, self.hand_l_actual, self.target_l, self.traj_l)
            for i, vi in enumerate(self.left_v_indices): dq[vi] = dq_l[i]
            dq_r, self.target_r = self._ik_step(self.right_hand_id, self.right_v_indices, self.hand_r_actual, self.target_r, self.traj_r)
            for i, vi in enumerate(self.right_v_indices): dq[vi] = dq_r[i]
            self.q_math = pin.integrate(self.model, self.q_math, dq * DT)

        for q_idx, g1_idx in self.pin_to_g1_q.items():
            cmd.motor_cmd[g1_idx].q, cmd.motor_cmd[g1_idx].dq, cmd.motor_cmd[g1_idx].tau, cmd.motor_cmd[g1_idx].kp, cmd.motor_cmd[g1_idx].kd = float(self.q_math[q_idx]), 0.0, 0.0, KP, KD

        for wi in G1_WAIST:
        cmd.motor_cmd[wi].q = 0.0 # Siempre recto
        cmd.motor_cmd[wi].kp = KP * 2.0
        cmd.motor_cmd[wi].kd = KD * 2.0

        cmd.motor_cmd[NOT_USED_JOINT].q = 1.0
        self.cmd_pub.publish(cmd)

    def release(self):
        cmd = LowCmd()
        cmd.motor_cmd[NOT_USED_JOINT].q = 0.0
        self.cmd_pub.publish(cmd)
        time.sleep(0.3)

# ──────────────────────────────────────────────────────────────────────────────
# 5.  MÁQUINA DE ESTADOS
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── Memoria compartida entre procesos ──
    manager = multiprocessing.Manager()
    shared_data = manager.dict()
    shared_data['detected'] = False
    shared_data['estado'] = "ESPERANDO ROBOT..."
    shared_data['mano_l'] = None
    shared_data['mano_r'] = None
    shared_data['limite_z'] = -999.0

    visión_proc = VisionProcess(shared_data)
    visión_proc.start()

    rclpy.init()
    robot = BimanalIK()
    spin_thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    spin_thread.start()

    print("[MAIN] Esperando estado del robot...")
    while not robot.state_received:
        time.sleep(0.1)
    print("[MAIN] Estado recibido. Iniciando máquina de estados.")

    estado           = "ANALIZANDO_ESCENA"
    tiempo_estado    = time.time()
    memoria_caja     = {}
    LIMITE_Z_SEGURO  = -999.0
    z_descenso_actual  = 0.0
    z_descenso_obj     = 0.0
    VELOCIDAD_DESCENSO = 0.006

    try:
        while True:
            t_loop = time.time()
            now = t_loop

            # Publicar estado visual a la cámara
            shared_data['estado'] = estado
            if robot.state_received:
                shared_data['mano_l'] = robot.hand_l_actual
                shared_data['mano_r'] = robot.hand_r_actual
            shared_data['limite_z'] = LIMITE_Z_SEGURO

            if estado == "ANALIZANDO_ESCENA":
                if shared_data.get('detected', False):
                    centro = shared_data.get('center')
                    z_mesa = shared_data.get('z_mesa')
                    if centro is not None and z_mesa is not None:
                        LIMITE_Z_SEGURO = z_mesa + 0.05
                        memoria_caja = {
                            'centro': centro.copy(),
                            'z_mesa': z_mesa,
                            'width':  max(shared_data.get('width', 0), ANCHO_CAJA_REAL),
                            'height': shared_data.get('height', 0),
                            'edge_l': shared_data.get('edge_l'),
                            'edge_r': shared_data.get('edge_r'),
                        }
                        print(f"[ESCENA] Centro caja: {centro} | Z mesa: {z_mesa:.3f}")
                        robot.active_ik = False
                        robot.sync_with_reality()
                        estado, tiempo_estado = "ABRIR_BRAZOS", now

            elif estado == "ABRIR_BRAZOS":
                if not hasattr(robot, '_base_roll_l'):
                    robot._base_roll_l = robot.q_math[robot.q_idx_l_roll]
                    robot._base_roll_r = robot.q_math[robot.q_idx_r_roll]
                progreso = min(1.0, (now - tiempo_estado) / 1.2)
                robot.q_math[robot.q_idx_l_roll] = robot._base_roll_l + 0.6 * progreso
                robot.q_math[robot.q_idx_r_roll] = robot._base_roll_r - 0.6 * progreso
                if progreso >= 1.0:
                    estado, tiempo_estado = "CALCULAR_PREPARAR", now

            elif estado == "CALCULAR_PREPARAR":
                centro = memoria_caja['centro']
                medio_ancho = memoria_caja['width'] / 2.0
                target_z = max(memoria_caja['z_mesa'] + 0.15, LIMITE_Z_SEGURO + 0.05)
                tgt_l = np.array([centro[0] - 0.10, centro[1] + medio_ancho + 0.15, target_z])
                tgt_r = np.array([centro[0] - 0.10, centro[1] - medio_ancho - 0.15, target_z])
                robot.lock_shoulder_roll, robot.lock_elbows_wrists, robot.use_6d = True, False, False
                robot.set_targets(tgt_l, tgt_r)
                estado, tiempo_estado = "MOVIENDO_PREPARAR", now

            elif estado == "MOVIENDO_PREPARAR":
                if robot.get_max_error() < 0.04 or (now - tiempo_estado > 5.0):
                    estado, tiempo_estado = "POSICION_SOBRE", now

            elif estado == "POSICION_SOBRE":
                centro = memoria_caja['centro']
                medio_ancho = memoria_caja['width'] / 2.0
                z_sobre = max(LIMITE_Z_SEGURO + 0.18, centro[2] + 0.12)
                tgt_x = centro[0] + (PROFUNDIDAD_CAJA_REAL / 2.0) - LONGITUD_MANO
                tgt_l = np.array([tgt_x, centro[1] + medio_ancho + 0.15, z_sobre])
                tgt_r = np.array([tgt_x, centro[1] - medio_ancho - 0.15, z_sobre])
                robot.lock_shoulder_roll, robot.use_6d = True, False
                robot.set_targets(tgt_l, tgt_r)
                estado, tiempo_estado = "MOVIENDO_SOBRE", now

            elif estado == "MOVIENDO_SOBRE":
                if robot.get_max_error() < 0.04 or (now - tiempo_estado > 5.0):
                    estado, tiempo_estado = "ALINEAR_PALMAS", now

            elif estado == "ALINEAR_PALMAS":
                pin.forwardKinematics(robot.model, robot.data, robot.q_math)
                pin.updateFramePlacements(robot.model, robot.data)
                ang = 0.8
                R_l = np.array([[1, 0, 0], [0, math.cos(-ang), -math.sin(-ang)], [0, math.sin(-ang), math.cos(-ang)]])
                R_r = np.array([[1, 0, 0], [0, math.cos(ang), -math.sin(ang)], [0, math.sin(ang), math.cos(ang)]])
                robot.target_rot_l = R_l @ robot.data.oMf[robot.left_hand_id].rotation.copy()
                robot.target_rot_r = R_r @ robot.data.oMf[robot.right_hand_id].rotation.copy()
                robot.use_6d, robot.lock_shoulder_roll = True, True

                centro = memoria_caja['centro']
                medio_ancho = memoria_caja['width'] / 2.0
                z_sobre = max(LIMITE_Z_SEGURO + 0.18, centro[2] + 0.12)
                tgt_x = centro[0] + (PROFUNDIDAD_CAJA_REAL / 2.0) - LONGITUD_MANO
                robot.set_targets(np.array([tgt_x, centro[1] + medio_ancho + 0.15, z_sobre]), 
                                  np.array([tgt_x, centro[1] - medio_ancho - 0.15, z_sobre]))
                estado, tiempo_estado = "MOVIENDO_ALINEAR", now

            elif estado == "MOVIENDO_ALINEAR":
                if robot.get_max_error() < 0.05 or (now - tiempo_estado > 4.0):
                    z_descenso_actual = float(robot.hand_l_actual[2])
                    z_descenso_obj = max((memoria_caja['centro'][2] + memoria_caja['z_mesa']) / 2.0, LIMITE_Z_SEGURO)
                    estado, tiempo_estado = "BAJAR_MANOS", now

            elif estado == "BAJAR_MANOS":
                robot.lock_shoulder_roll = False
                z_descenso_actual = z_descenso_actual - VELOCIDAD_DESCENSO if (z_descenso_actual - VELOCIDAD_DESCENSO) >= z_descenso_obj else z_descenso_obj
                centro = memoria_caja['centro']
                medio_ancho = memoria_caja['width'] / 2.0
                tgt_x = centro[0] + (PROFUNDIDAD_CAJA_REAL / 2.0) - LONGITUD_MANO

                if not robot.traj_l and not robot.traj_r:
                    robot.set_targets(np.array([tgt_x, centro[1] + medio_ancho + 0.15, z_descenso_actual]), 
                                      np.array([tgt_x, centro[1] - medio_ancho - 0.15, z_descenso_actual]))

                if z_descenso_actual <= z_descenso_obj + 0.005 or now - tiempo_estado > 7.0:
                    estado, tiempo_estado = "CERRAR_AGARRE", now

            elif estado == "CERRAR_AGARRE":
                robot.lock_shoulder_roll = False
                centro = memoria_caja['centro']
                borde_l = centro[1] + (memoria_caja['width'] / 2.0) - 0.045
                borde_r = centro[1] - (memoria_caja['width'] / 2.0) + 0.045
                dist_l = float(robot.hand_l_actual[1]) - borde_l
                dist_r = borde_r - float(robot.hand_r_actual[1])

                if (dist_l < 0.01 and dist_r < 0.01) or (now - tiempo_estado > 16.0):
                    estado, tiempo_estado = "LEVANTAR_RETRAER", now
                elif not robot.traj_l and not robot.traj_r:
                    paso_l, paso_r = min(0.01, max(0.0, dist_l)), min(0.01, max(0.0, dist_r))
                    tgt_x = centro[0] + (PROFUNDIDAD_CAJA_REAL / 2.0) - LONGITUD_MANO
                    robot.set_targets(np.array([tgt_x, robot.hand_l_actual[1] - paso_l, z_descenso_obj]), 
                                      np.array([tgt_x, robot.hand_r_actual[1] + paso_r, z_descenso_obj]))

            elif estado == "LEVANTAR_RETRAER":
                robot.lock_shoulder_roll = True
                tgt_l, tgt_r = robot.hand_l_actual.copy(), robot.hand_r_actual.copy()
                tgt_l[0] = 0.20; tgt_r[0] = 0.20
                tgt_l[2] += 0.05; tgt_r[2] += 0.05
                robot.set_targets(tgt_l, tgt_r)
                estado, tiempo_estado = "MOVIENDO_RETRAER", now

            elif estado == "MOVIENDO_RETRAER":
                if robot.get_max_error() < 0.04 or (now - tiempo_estado > 4.0):
                    estado, tiempo_estado = "LEVANTAR_SUBIR", now

            elif estado == "LEVANTAR_SUBIR":
                tgt_l, tgt_r = robot.hand_l_actual.copy(), robot.hand_r_actual.copy()
                tgt_l[2] = 0.40; tgt_r[2] = 0.40
                robot.set_targets(tgt_l, tgt_r)
                estado, tiempo_estado = "MOVIENDO_SUBIR", now

            elif estado == "MOVIENDO_SUBIR":
                if robot.get_max_error() < 0.04 or (now - tiempo_estado > 5.0):
                    print("[ESTADO] ✅ Caja levantada con éxito.")
                    estado, tiempo_estado = "FINALIZADO", now

            time.sleep(max(0.0, DT - (time.time() - t_loop)))

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupción de teclado.")

    finally:
        robot.release()
        visión_proc.terminate()
        visión_proc.join()
        robot.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
