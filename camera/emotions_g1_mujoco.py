import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import time
import threading
import numpy as np
import pinocchio as pin
import json
import os
import socket

class G1SimulatedEmotions(Node):
    def __init__(self):
        super().__init__('g1_simulated_emotions')
        
        self.dt = 0.02
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Suscribirse al topic que genera tu script puente
        self.state_sub = self.create_subscription(JointState, '/joint_states', self.state_callback, 10)
        
        self.state_received = False
        
        self.g1_arm_left = [15, 16, 17, 18, 19, 20, 21]
        self.g1_arm_right = [22, 23, 24, 25, 26, 27, 28] 
        self.current_jpos = [0.0] * 29 
        
        self.active_ik = False       
        self.trajectory_q = [] 
        
        # Nombres exactos para mapear el JointState a índices 0-28
        self.joint_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]

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
        except Exception as e:
            print(f"[ERROR] Error cargando URDF: {e}")
            self.model = None

        self.arm_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
        ]

        self.pin_to_g1_q = {}
        if self.model is not None:
            for i, name in enumerate(self.arm_names):
                if self.model.existJointName(name):
                    j_id = self.model.getJointId(name)
                    q_idx = self.model.joints[j_id].idx_q
                    self.pin_to_g1_q[q_idx] = self.g1_arm_left[i]

        self.q_math = pin.neutral(self.model)
        self.home_q_math = None
        self.timer = self.create_timer(self.dt, self.control_loop)
        print("[INFO] Esperando estado de MuJoCo...")

    def state_callback(self, msg):
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                self.current_jpos[idx] = msg.position[i]
                
        if not self.state_received:
            self.state_received = True
            self.sync_math_with_reality()
            self.home_q_math = self.q_math.copy()
            print("[INFO] Posición natural capturada. Emociones listas para Simulación.")
            threading.Thread(target=self.udp_listener_loop, daemon=True).start()

    def sync_math_with_reality(self):
        for q_idx, g1_idx in self.pin_to_g1_q.items():
            self.q_math[q_idx] = self.current_jpos[g1_idx]

    def move_to_pose(self, target_q_left, duration=2.0):
        num_steps = max(1, int(duration / self.dt))
        start_q = self.q_math.copy()
        target_q_math = start_q.copy()
        for i, name in enumerate(self.arm_names):
            j_id = self.model.getJointId(name)
            q_idx = self.model.joints[j_id].idx_q
            target_q_math[q_idx] = target_q_left[i]
        self.trajectory_q = [start_q + (i / num_steps) * (target_q_math - start_q) for i in range(1, num_steps + 1)]
        self.active_ik = True

    def move_to_home(self, duration=1.0): 
        num_steps = max(1, int(duration / self.dt))
        start_q = self.q_math.copy()
        self.trajectory_q = [start_q + (i / num_steps) * (self.home_q_math - start_q) for i in range(1, num_steps + 1)]
        self.active_ik = True

    def wait_until_reached(self):
        while self.active_ik and self.trajectory_q:
            time.sleep(0.02)

    def release_control(self):
        self.active_ik = False
        self.trajectory_q = []
        # Enviar JSON vacío suelta los brazos en tu script principal
        self.udp_sock.sendto(b"{}", ('127.0.0.1', 9876))
        print("[INFO] Emoción finalizada. Control devuelto a ONNX.")

    def play_emotion(self, emotion):
        print(f"\n🎭 Reproduciendo emoción en SIMULACIÓN: {emotion}")
        self.sync_math_with_reality() 
        
        if emotion == "HAPPY":
            pose_up = [0.0, 0.3, 0.2, -1.0, 0.0, 0.0, 0.0]
            pose_next = [-2.8, 0.3, 0.4, -1.0, 0.0, 0.0, 0.0]
            self.move_to_pose(pose_up, duration=1.0)
            self.wait_until_reached()
            self.move_to_pose(pose_next, duration=1.5)
            self.wait_until_reached()
            self.move_to_home(duration=1.3)
            self.wait_until_reached()
            self.release_control()
            
        elif emotion == "SAD":
            pose_up = [0.0, 0.0, 0.0, -1.0, 0.0, -0.0, 0.0]
            pose_frust = [-1.0, 1.0, -0.2, -0.8, 0.9, -0.0, 0.5]
            self.move_to_pose(pose_up, duration=1.2)
            self.wait_until_reached()
            self.move_to_pose(pose_frust, duration=1.2)
            self.wait_until_reached()
            time.sleep(2.0)
            self.move_to_home(duration=1.5)
            self.wait_until_reached()
            self.release_control()

        # Añade el resto de emociones (ANGRY, DANCE, FRUSTRATED) exactamente igual que en el original
        else:
            print(f"[WARN] Emoción {emotion} no implementada en este snippet.")
            self.release_control()

    def udp_listener_loop(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        server_socket.bind(('0.0.0.0', 5005))
        print("[INFO] 🎧 Escuchando emociones UDP en el puerto 5005...")
        while True:
            try:
                data, _ = server_socket.recvfrom(1024)
                cmd = data.decode('utf-8').strip().upper()
                if cmd in ["HAPPY", "NEUTRAL", "FRUSTRATED", "SAD", "ANGRY", "DANCE"]:
                    self.play_emotion(cmd)
            except: pass

    def control_loop(self):
        if not self.state_received or self.model is None or not self.active_ik: 
            return

        if self.trajectory_q:
            self.q_math = self.trajectory_q.pop(0)

        comandos_brazos = {}
        for q_idx, g1_idx in self.pin_to_g1_q.items():
            comandos_brazos[g1_idx] = float(self.q_math[q_idx])
            
        target_dict = {}
        for i in range(7):
            left_motor_id = self.g1_arm_left[i]
            right_motor_id = self.g1_arm_right[i]
            left_angle = comandos_brazos[left_motor_id]
            right_angle = -left_angle if i in [1, 2, 4, 6] else left_angle
            
            target_dict[left_motor_id] = left_angle
            target_dict[right_motor_id] = right_angle

        # Enviar JSON al script principal (ONNX) para que sobrescriba la locomoción
        self.udp_sock.sendto(json.dumps(target_dict).encode('utf-8'), ('127.0.0.1', 9876))

def main(args=None):
    rclpy.init(args=args)
    node = G1SimulatedEmotions()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
