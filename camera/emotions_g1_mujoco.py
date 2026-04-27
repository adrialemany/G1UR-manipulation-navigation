import os
import sys
import time
import math
import numpy as np
import pinocchio as pin
import json
import socket
import threading

os.environ["CYCLONEDDS_URI"] = """<CycloneDDS>
    <Domain>
        <SharedMemory>
            <Enable>false</Enable>
        </SharedMemory>
    </Domain>
</CycloneDDS>"""

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

def send_walk_cmd(cmd):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 6000))
        s.sendall(cmd.encode('utf-8'))
        s.recv(1024)
        s.close()
    except: pass

class G1PerfectIK:
    def __init__(self):
        self.NOT_USED_JOINT = 29 
        self.dt = 0.02
        
        self.g1_arm_left = [15, 16, 17, 18, 19, 20, 21]
        self.g1_arm_right = [22, 23, 24, 25, 26, 27, 28] 
        self.g1_waist = [12, 13, 14]
        self.current_jpos = [0.0] * 29 
        
        self.active_ik = False       
        self.trajectory_points = []
        self.trajectory_q = [] 
        self.current_target_xyz = None
        self.final_target_xyz = None
        
        self.hand_xyz_actual = np.zeros(3)
        self.home_xyz = None
        self.home_q_math = None
        self.wrist_roll_offset = 0.0
        self.current_wrist_offset = 0.0 
        
        self.safe_zone = self.load_safe_zone("left_arm_safe_zone.json")
        
        self.joint_safety_limits = {
            15: (-3.04, 2.62), 16: (-1.54, 2.20), 17: (-2.57, 2.57),
            18: (-1.00, 2.04), 19: (-1.92, 1.92), 20: (-1.56, 1.56),
            21: (-1.56, 1.56)
        }
        
        urdf_path = os.path.expanduser("~/robot_ws/src/g1pilot/description_files/urdf/g1_29dof.urdf")
        try:
            self.full_model = pin.buildModelFromUrdf(urdf_path)
            joints_to_lock_names = [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
                "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
            ]
            locked_joint_ids = [self.full_model.getJointId(j) for j in joints_to_lock_names if self.full_model.existJointName(j)]
            q_neutral = pin.neutral(self.full_model)
            
            self.model = pin.buildReducedModel(self.full_model, locked_joint_ids, q_neutral)
            self.data = self.model.createData()
            self.hand_frame_id = self.model.getFrameId("left_rubber_hand")
        except Exception as e:
            print(f"Error cargando URDF: {e}")
            self.model = None

        self.arm_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
        ]

        self.pin_to_g1_q = {}
        self.arm_v_indices = []
        
        if self.model is not None:
            for i, name in enumerate(self.arm_names):
                if self.model.existJointName(name):
                    j_id = self.model.getJointId(name)
                    q_idx = self.model.joints[j_id].idx_q
                    v_idx = self.model.joints[j_id].idx_v
                    
                    self.pin_to_g1_q[q_idx] = self.g1_arm_left[i]
                    self.arm_v_indices.append(v_idx)

        self.q_math = pin.neutral(self.model)
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_address = ('127.0.0.1', 9876)
        self.state_received = False
        self.tick_count = 0
        
        ChannelFactoryInitialize(1, "lo") 
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.state_callback, 10)
        
        print("[INFO] Esperando estado de la simulación...")

    def load_safe_zone(self, file_path):
        if not os.path.exists(file_path):
            print(f"Aviso: No hay {file_path}. Usando límites genéricos.")
            return {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0, 'z_min': -1.0, 'z_max': 1.0}
        with open(file_path, 'r') as f:
            zone = json.load(f)
        print("✅ Zona segura cargada.")
        return zone

    def state_callback(self, msg: LowState_):
        self.tick_count += 1
        if self.tick_count < 150: 
            return
            
        for i in range(29):
            self.current_jpos[i] = msg.motor_state[i].q
            
        if not self.state_received:
            self.state_received = True
            self.sync_math_with_reality()
            
            self.home_xyz = self.hand_xyz_actual.copy()
            self.home_q_math = self.q_math.copy()
            
            self.active_ik = True
            self.current_target_xyz = self.home_xyz.copy()
            
            print("[INFO] Posición natural capturada. Emociones listas.")
            threading.Thread(target=self.control_loop, daemon=True).start()
            threading.Thread(target=self.udp_listener_loop, daemon=True).start()
            
    def sync_math_with_reality(self):
        for q_idx, g1_idx in self.pin_to_g1_q.items():
            self.q_math[q_idx] = self.current_jpos[g1_idx]
        pin.forwardKinematics(self.model, self.data, self.q_math)
        pin.updateFramePlacements(self.model, self.data)
        self.hand_xyz_actual = self.data.oMf[self.hand_frame_id].translation.copy()

    def check_joint_limits(self, q_state):
        for q_idx, g1_idx in self.pin_to_g1_q.items():
            if g1_idx in self.joint_safety_limits:
                min_q, max_q = self.joint_safety_limits[g1_idx]
                val = q_state[q_idx]
                if val < min_q or val > max_q:
                    return False
        return True

    def clamp_target_to_safe_zone(self, start_xyz, target_xyz):
        sz = self.safe_zone
        if (sz['x_min'] <= target_xyz[0] <= sz['x_max'] and
            sz['y_min'] <= target_xyz[1] <= sz['y_max'] and
            sz['z_min'] <= target_xyz[2] <= sz['z_max']):
            return target_xyz

        direction = target_xyz - start_xyz
        direction[direction == 0] = 1e-6
        
        t_x_min, t_x_max = (sz['x_min'] - start_xyz[0]) / direction[0], (sz['x_max'] - start_xyz[0]) / direction[0]
        t_y_min, t_y_max = (sz['y_min'] - start_xyz[1]) / direction[1], (sz['y_max'] - start_xyz[1]) / direction[1]
        t_z_min, t_z_max = (sz['z_min'] - start_xyz[2]) / direction[2], (sz['z_max'] - start_xyz[2]) / direction[2]
        
        t_enter = max(min(t_x_min, t_x_max), min(t_y_min, t_y_max), min(t_z_min, t_z_max))
        t_exit = min(max(t_x_min, t_x_max), max(t_y_min, t_y_max), max(t_z_min, t_z_max))
        
        t_safe = max(0.0, max(0.0, min(1.0, t_exit)) - 0.001)
        return start_xyz + direction * t_safe

    def move_to(self, target_xyz, wrist_rot=0.0, duration=0.4):
        target_xyz = np.array(target_xyz)
        start_pos = self.current_target_xyz if (self.active_ik and self.current_target_xyz is not None) else self.hand_xyz_actual
        
        safe_target = self.clamp_target_to_safe_zone(self.hand_xyz_actual, target_xyz)
        self.final_target_xyz = safe_target
        
        num_steps = max(1, int(duration / self.dt))
        self.trajectory_points = [start_pos + (i / num_steps) * (safe_target - start_pos) for i in range(1, num_steps + 1)]
        
        self.wrist_trajectory = np.linspace(self.current_wrist_offset, wrist_rot, num_steps).tolist()
        self.wrist_roll_offset = wrist_rot 
        
        self.trajectory_q = [] 
        self.active_ik = True

    def move_to_pose(self, target_q_left, duration=2.0):
        num_steps = max(1, int(duration / self.dt))
        start_q = self.q_math.copy()
        target_q_math = start_q.copy()
        
        for i, name in enumerate(self.arm_names):
            j_id = self.model.getJointId(name)
            q_idx = self.model.joints[j_id].idx_q
            target_q_math[q_idx] = target_q_left[i]
            
        self.trajectory_q = [start_q + (i / num_steps) * (target_q_math - start_q) for i in range(1, num_steps + 1)]
        
        self.wrist_trajectory = np.linspace(self.current_wrist_offset, 0.0, num_steps).tolist()
        self.wrist_roll_offset = 0.0
        self.trajectory_points = [] 
        
        self.current_target_xyz = None 
        self.final_target_xyz = self.home_xyz.copy() 
        self.active_ik = True

    def move_to_home(self, duration=1.0): 
        self.final_target_xyz = self.home_xyz.copy()
        self.current_target_xyz = self.home_xyz.copy() 
        
        num_steps = max(1, int(duration / self.dt))
        start_q = self.q_math.copy()
        
        self.trajectory_q = [start_q + (i / num_steps) * (self.home_q_math - start_q) for i in range(1, num_steps + 1)]
        
        self.wrist_trajectory = np.linspace(self.current_wrist_offset, 0.0, num_steps).tolist()
        self.wrist_roll_offset = 0.0
        
        self.trajectory_points = [] 
        self.active_ik = True

    def wait_until_reached(self):
        t0 = time.time()
        while True:
            if not self.active_ik: 
                break
                
            if self.trajectory_q or self.trajectory_points:
                time.sleep(0.02)
                t0 = time.time() 
                continue
                
            if self.final_target_xyz is not None and self.home_xyz is not None and not np.allclose(self.final_target_xyz, self.home_xyz):
                err = self.final_target_xyz - self.hand_xyz_actual
                if np.linalg.norm(err) < 0.05: 
                    break
                if time.time() - t0 > 2.0: 
                    break
            else:
                break
                
            time.sleep(0.02)

    def play_emotion(self, emotion):
        print(f"\n🎭 Reproduciendo emoción: {emotion}")
        
        if emotion == "HAPPY":
            # Restaurado a XYZ con duraciones ajustadas a la fluidez
            self.move_to([0, 0.45, 0.53], duration=0.4)
            self.wait_until_reached()
            self.move_to([0, 0.26, 0.63], duration=0.4)
            self.wait_until_reached()
            self.move_to([0, 0.45, 0.53], duration=0.4)
            self.wait_until_reached()
            self.move_to([0, 0.26, 0.63], duration=0.4)
            self.wait_until_reached()
            self.move_to_home(duration=1.0)
            self.wait_until_reached()
            
        elif emotion == "NEUTRAL":
            pose_izq = [0.6, 0.3, 0.9, -1.5, -1.8, -0.2, 1.0]
            self.move_to_pose(pose_izq, duration=1.5)
            self.wait_until_reached()
            
            time.sleep(3.0)
            self.move_to_home(duration=1.5)
            self.wait_until_reached()
            
        elif emotion == "FRUSTRATED":
            # Restaurado a XYZ
            self.move_to([0.1, 0.17, 0.4], duration=0.8) 
            self.wait_until_reached()
            time.sleep(4.0) 
            self.move_to_home(duration=1.0)
            self.wait_until_reached()
            
        elif emotion == "SAD":
            # Restaurado a XYZ
            self.move_to([0.15, 0.06, 0.4], duration=0.8)
            self.wait_until_reached()
            time.sleep(4.0) 
            self.move_to_home(duration=1.0)
            self.wait_until_reached()
            
        elif emotion == "ANGRY":
            send_walk_cmd('w')
            time.sleep(0.5)
            send_walk_cmd('stop')
            
            # 1. PRIMERO: Llegar progresivamente a la postura base de enfado (guardia alta)
            # Hombros hacia adelante (-0.8), un poco abiertos (0.3), y codo medio doblado (0.5)
            pose_guardia = [-0.8, 0.3, 0.0, -0.4, 0.0, 0.0, 0.0]
            self.move_to_pose(pose_guardia, duration=0.8)
            self.wait_until_reached()
            
            # 2. Creamos las poses de "bombeo" basándonos EXACTAMENTE en esa guardia,
            # alterando únicamente el valor del codo (índice 3 de la lista).
            pose_codo_flexionado = list(pose_guardia)
            pose_codo_flexionado[3] = -1.0  # Dobla el codo hacia arriba (el límite es 2.04)
            
            pose_codo_extendido = list(pose_guardia)
            pose_codo_extendido[3] = -0.4   # Extiende el codo (vuelve a la guardia)

            # 3. Ejecutamos el bucle "bombeando" solo el codo
            for _ in range(3):
                self.move_to_pose(pose_codo_flexionado, duration=0.5)
                self.wait_until_reached()
                
                self.move_to_pose(pose_codo_extendido, duration=0.5)
                self.wait_until_reached()
                
            # 4. Finalizamos volviendo a casa
            self.move_to_home(duration=1.0)
            self.wait_until_reached()
            
            send_walk_cmd('s')
            time.sleep(0.5)
            send_walk_cmd('stop')
            
        elif emotion == "DANCE":
            # ── COREOGRAFÍA SIMULADA ──
            # Mezclamos comandos de locomoción (piernas) con cinemática inversa (brazos)
            
            pose_arriba = [-1.5, 0.5, 0.0, -1.5, 0.0, 0.0, 0.0]  # Brazos bien altos
            pose_abajo = [0.2, 0.6, 0.0, -0.5, 1.57, 0.0, 0.0]   # Brazos bajos y palmas arriba
            pose_giro = [-0.2, 1.2, 0.0, -2.0, 0.0, -0.5, 0.0]   # Pose de "gimnasta" para el giro
            
            # 1. Bucle de "Groove" (Paso a la izquierda y a la derecha)
            for _ in range(2):
                # Paso lateral izquierda + Brazos arriba
                send_walk_cmd('a') 
                self.move_to_pose(pose_arriba, duration=0.6)
                self.wait_until_reached()
                
                # Paso lateral derecha + Brazos abajo
                send_walk_cmd('d')
                self.move_to_pose(pose_abajo, duration=0.6)
                self.wait_until_reached()

            # 2. El Giro Final (Spin)
            send_walk_cmd('stop')
            time.sleep(0.1)
            send_walk_cmd('q') # Comenzar a rotar sobre sí mismo
            self.move_to_pose(pose_giro, duration=1.2)
            self.wait_until_reached()
            
            # 3. Terminar con estilo y volver a casa
            send_walk_cmd('stop')
            self.move_to_home(duration=1.0)
            self.wait_until_reached()

    def udp_listener_loop(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        try:
            server_socket.bind(('0.0.0.0', 5005))
            print("[INFO] 🎧 Escuchando emociones por red en el puerto 5005...")
        except Exception as e:
            print(f"[ERROR] No se pudo hacer bind en el puerto 5005: {e}")
            return
            
        while True:
            try:
                data, addr = server_socket.recvfrom(1024)
                cmd = data.decode('utf-8').strip().upper()
                if cmd in ["HAPPY", "NEUTRAL", "FRUSTRATED", "SAD", "ANGRY", "DANCE"]:
                    print(f"📥 Emoción recibida desde {addr}: {cmd}")
                    self.play_emotion(cmd)
            except Exception as e:
                print(f"Error recibiendo comando: {e}")

    def control_loop(self):
        while True:
            t_start = time.time()
            if not self.state_received or self.model is None: 
                time.sleep(self.dt)
                continue

            if not self.active_ik:
                try:
                    self.udp_sock.sendto(json.dumps({}).encode('utf-8'), self.target_address)
                except Exception:
                    pass
                time.sleep(self.dt)
                continue
                
            if hasattr(self, 'wrist_trajectory') and self.wrist_trajectory:
                self.current_wrist_offset = self.wrist_trajectory.pop(0)

            # MODO 1: POSE DIRECTA O HOME (Totalmente suave, sin fallos)
            if self.trajectory_q:
                self.q_math = self.trajectory_q.pop(0)
                pin.forwardKinematics(self.model, self.data, self.q_math)
                pin.updateFramePlacements(self.model, self.data)
                self.hand_xyz_actual = self.data.oMf[self.hand_frame_id].translation
                
            # MODO 2: CARTESIAN IK (Para emociones basadas en XYZ)
            elif self.current_target_xyz is not None:
                if self.trajectory_points:
                    self.current_target_xyz = self.trajectory_points.pop(0)
                
                pin.forwardKinematics(self.model, self.data, self.q_math)
                pin.updateFramePlacements(self.model, self.data)
                self.hand_xyz_actual = self.data.oMf[self.hand_frame_id].translation

                err = self.current_target_xyz - self.hand_xyz_actual
                err_norm = np.linalg.norm(err)
                
                max_step = 0.1
                if err_norm > max_step: err = (err / err_norm) * max_step

                dq = np.zeros(self.model.nv)
                
                if err_norm > 0.002:
                    J = pin.computeFrameJacobian(self.model, self.data, self.q_math, self.hand_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    J_pos = J[:3, :]
                    J_arm = J_pos[:, self.arm_v_indices] 
                    
                    lambda_dls = 0.05
                    pseudo_inv = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + (lambda_dls**2) * np.eye(3))
                    
                    dq_primary = pseudo_inv @ err * 4.0 
                    
                    dq_secondary = np.zeros(7)
                    k_repulsion = 4.0 
                    margin = 0.4 
                    
                    for i, name in enumerate(self.arm_names):
                        j_id = self.model.getJointId(name)
                        q_idx = self.model.joints[j_id].idx_q
                        g1_idx = self.g1_arm_left[i]

                        if g1_idx in self.joint_safety_limits:
                            min_q, max_q = self.joint_safety_limits[g1_idx]
                            current_q = self.q_math[q_idx]
                            
                            if current_q < (min_q + margin):
                                dq_secondary[i] = ((min_q + margin) - current_q) * k_repulsion
                            elif current_q > (max_q - margin):
                                dq_secondary[i] = ((max_q - margin) - current_q) * k_repulsion

                    I = np.eye(7)
                    null_space_projector = I - (pseudo_inv @ J_arm)
                    dq_arm = dq_primary + (null_space_projector @ dq_secondary)

                    for i, v_idx in enumerate(self.arm_v_indices):
                        dq[v_idx] = dq_arm[i]

                # ── PROTECCIÓN FÍSICA ANTI-TIRONES VECTORIAL ──
                # Mantiene la dirección de la ruta recta, escalando el vector completo 
                # en lugar de "cortar" ejes individuales.
                max_vel = 8.0 
                max_dq = np.max(np.abs(dq))
                if max_dq > max_vel:
                    dq = dq * (max_vel / max_dq)

                q_math_future = pin.integrate(self.model, self.q_math, dq * self.dt)

                # ── LIMITADOR ARTICULAR SUAVE ──
                # Si llega al límite, se desliza por el borde en lugar de abortar la memoria.
                for q_idx, g1_idx in self.pin_to_g1_q.items():
                    if g1_idx in self.joint_safety_limits:
                        min_q, max_q = self.joint_safety_limits[g1_idx]
                        q_math_future[q_idx] = max(min_q, min(max_q, q_math_future[q_idx]))
                        
                self.q_math = q_math_future

            comandos_brazos = {}
            for q_idx, g1_idx in self.pin_to_g1_q.items():
                comandos_brazos[g1_idx] = float(self.q_math[q_idx])
            
            for i in range(7):
                left_motor_id = self.g1_arm_left[i]
                right_motor_id = self.g1_arm_right[i]
                left_angle = comandos_brazos[left_motor_id]
                
                if i == 4: 
                    left_angle += self.current_wrist_offset
                    min_q, max_q = self.joint_safety_limits[19]
                    left_angle = max(min_q, min(max_q, left_angle))
                    
                if i in [1, 2, 4, 6]:
                    comandos_brazos[right_motor_id] = -left_angle
                else:
                    comandos_brazos[right_motor_id] = left_angle
            
            try:
                self.udp_sock.sendto(json.dumps(comandos_brazos).encode('utf-8'), self.target_address)
            except Exception:
                pass

            time.sleep(max(0.0, self.dt - (time.time() - t_start)))

if __name__ == '__main__':
    node = G1PerfectIK()
    try:
        while not node.state_received or node.home_xyz is None:
            time.sleep(0.1)
            
        print("\n" + "="*50)
        print("🤖 CONTROL DE EMOCIONES INICIADO 🤖")
        print("="*50)
        
        while True:
            cmd = input("\n> Ingresa emoción (HAPPY, NEUTRAL, FRUSTRATED, SAD, ANGRY, DANCE) o 'q' para salir: ").strip().upper()
            if cmd == 'Q':
                break
            elif cmd in ["HAPPY", "NEUTRAL", "FRUSTRATED", "SAD", "ANGRY", "DANCE"]:
                node.play_emotion(cmd)
            else:
                print("[ERROR] Comando no reconocido. Inténtalo de nuevo.")
                
    except KeyboardInterrupt:
        print("\nSaliendo...")
        os._exit(0)
