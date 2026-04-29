import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from unitree_hg.msg import LowCmd, LowState
import time
import threading
import numpy as np
import pinocchio as pin
import json
import os
import socket
import sys

def send_walk_cmd(cmd):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 6000))
        s.sendall(cmd.encode('utf-8'))
        s.recv(1024)
        s.close()
    except: pass

class G1PhysicalEmotions(Node):
    def __init__(self):
        super().__init__('g1_physical_emotions')
        
        self.NOT_USED_JOINT = 29 
        self.kp = 60.0
        self.kd = 1.5
        self.dt = 0.02  # 50Hz
        
        self.cmd_pub = self.create_publisher(LowCmd, '/arm_sdk', 10)
        self.state_sub = self.create_subscription(LowState, '/lowstate', self.state_callback, qos_profile_sensor_data)
        
        self.low_state = None
        self.state_received = False
        self.tick_count = 0
        
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
        self.home_waist_jpos = [0.0, 0.0, 0.0] # <--- Guardarem la postura de la cintura
        self.wrist_roll_offset = 0.0
        self.current_wrist_offset = 0.0 
        
        self.safe_zone = self.load_safe_zone("left_arm_safe_zone.json")
        
        self.joint_safety_limits = {
            15: (-3.04, 2.62), 
            16: (-1.54, 2.20), 
            17: (-2.57, 2.57),
            18: (-1.00, 2.04), 
            19: (-1.92, 1.92), 
            20: (-1.56, 1.56),
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
            self.get_logger().error(f"Error cargant URDF: {e}")
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
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        print("[INFO] Esperant estat del robot físic...")

    def load_safe_zone(self, file_path):
        if not os.path.exists(file_path):
            return {'x_min': -1.0, 'x_max': 1.0, 'y_min': -1.0, 'y_max': 1.0, 'z_min': -1.0, 'z_max': 1.0}
        with open(file_path, 'r') as f:
            zone = json.load(f)
        return zone

    def state_callback(self, msg):
        self.low_state = msg
        for i in range(29):
            if i < len(self.low_state.motor_state):
                self.current_jpos[i] = self.low_state.motor_state[i].q
                
        if not self.state_received:
            self.state_received = True
            self.sync_math_with_reality()
            
            self.home_xyz = self.hand_xyz_actual.copy()
            self.home_q_math = self.q_math.copy()
            
            # <--- ACÍ CAPTUREM LA CINTURA RECTA --->
            self.home_waist_jpos = [self.current_jpos[w] for w in self.g1_waist]
            
            self.active_ik = False 
            
            print("[INFO] Posició natural capturada. Emocions llestes.")
            threading.Thread(target=self.udp_listener_loop, daemon=True).start()
            
    def sync_math_with_reality(self):
        for q_idx, g1_idx in self.pin_to_g1_q.items():
            self.q_math[q_idx] = self.current_jpos[g1_idx]
        pin.forwardKinematics(self.model, self.data, self.q_math)
        pin.updateFramePlacements(self.model, self.data)
        self.hand_xyz_actual = self.data.oMf[self.hand_frame_id].translation.copy()

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
        self.trajectory_points = []
        self.trajectory_q = []
        self.current_target_xyz = None
        
        cmd = LowCmd()
        # Enviar mode 0 garanteix que l'SDK solta els braços (i la cintura) per complet
        for j in range(29):
            cmd.motor_cmd[j].mode = 0
            
        cmd.motor_cmd[self.NOT_USED_JOINT].q = 0.0
        self.cmd_pub.publish(cmd)
        print("[INFO] Emoció finalitzada. Control retornat a la policy d'Unitree.")

    def play_emotion(self, emotion):
        print(f"\n🎭 Reproduint emoció: {emotion}")
        self.sync_math_with_reality() 
        
        if emotion == "HAPPY":
            pose_up = [0.0, 0.3, 0.2, -1.0, 0.0, 0.0, 0.0]
            pose_next = [-2.5, 0.3, 0.4, -1.0, 0.0, 0.0, 0.0]
            pose_next2 = [-2.5, 0.7, 0.0, 0.4, 0.0, 0.0, 0.0]
            pose_wave1 = [-2.5, 1.3, 0.0, 0.4, 0.0, 0.0, 0.0]
            pose_wave2 = [-2.5, 0.7, 0.0, 0.4, 0.0, 0.0, 0.0]
            
            self.move_to_pose(pose_up, duration=1.0)
            self.wait_until_reached()
            self.move_to_pose(pose_next, duration=1.5)
            self.wait_until_reached()
            self.move_to_pose(pose_next2, duration=1.5)
            self.wait_until_reached()


            
            for _ in range(2):
                self.move_to_pose(pose_wave1, duration=0.7)
                self.wait_until_reached()
                self.move_to_pose(pose_wave2, duration=0.7)
                self.wait_until_reached()
            
            self.move_to_pose(pose_next2, duration=1.0)
            self.wait_until_reached()
            self.move_to_pose(pose_next, duration=1.5)
            self.wait_until_reached()
            self.move_to_pose(pose_up, duration=1.5)
            self.wait_until_reached()


            self.move_to_home(duration=1.0)
            self.wait_until_reached()
            self.release_control()
            
        elif emotion == "NEUTRAL":
            pose_izq = [0.6, 0.3, 0.9, -1.5, -1.8, -0.2, 1.0]
            self.move_to_pose(pose_izq, duration=1.5)
            self.wait_until_reached()
            
            time.sleep(3.0)
            self.move_to_home(duration=1.5)
            self.wait_until_reached()
            self.release_control()
            
        elif emotion == "FRUSTRATED":
            pose_up = [0.0, 0.0, 0.0, -1.0, 0.0, -0.0, 0.0]
            pose_frust = [-1.5, 0.5, 0.0, -0.8, 0.0, -0.0, 0.0]
            self.move_to_pose(pose_up, duration=1.2)
            self.wait_until_reached()

            self.move_to_pose(pose_frust, duration=1.2)
            self.wait_until_reached()
            
            time.sleep(4.0)
            self.move_to_pose(pose_up, duration=1.2)
            self.wait_until_reached()
 
            self.move_to_home(duration=1.5)
            self.wait_until_reached()
            self.release_control()
            
        elif emotion == "SAD":
            pose_up = [0.0, 0.0, 0.0, -1.0, 0.0, -0.0, 0.0]
            pose_frust = [-1.0, 1.0, -0.2, -0.8, 0.7, -0.0, 0.0]
            self.move_to_pose(pose_up, duration=1.2)
            self.wait_until_reached()

            self.move_to_pose(pose_frust, duration=1.2)
            self.wait_until_reached()
            
            time.sleep(4.0)
            self.move_to_pose(pose_up, duration=1.2)
            self.wait_until_reached()
 
            self.move_to_home(duration=1.5)
            self.wait_until_reached()
            self.release_control()

        elif emotion == "ANGRY":
            send_walk_cmd('w')
            time.sleep(0.2)
            send_walk_cmd('stop')
            
            pose_guardia = [-0.8, 0.3, 0.0, -0.4, 0.0, 0.0, 0.0]
            self.move_to_pose(pose_guardia, duration=0.8)
            self.wait_until_reached()
            
            pose_codo_flexionado = list(pose_guardia)
            pose_codo_flexionado[3] = -1.0  
            pose_codo_extendido = list(pose_guardia)
            pose_codo_extendido[3] = -0.4   

            for _ in range(3):
                self.move_to_pose(pose_codo_flexionado, duration=0.4)
                self.wait_until_reached()
                self.move_to_pose(pose_codo_extendido, duration=0.4)
                self.wait_until_reached()
                
            self.move_to_home(duration=1.0)
            self.wait_until_reached()
            
            send_walk_cmd('s')
            time.sleep(0.2)
            send_walk_cmd('stop')
            self.release_control()
            
        elif emotion == "DANCE":
            pose_arriba = [-1.0, 0.5, -0.8, -0.8, 0.0, -0.0, 0.7]  
            pose_abajo = [-0.2, 0.1, 0.0, 0.8, 0.0, 0.0, 0.0] 
            pose_up = [0.0, 0.3, 0.2, -1.0, 0.0, 0.0, 0.0]
            pose_next = [-2.5, 0.3, 0.4, -1.0, 0.0, 0.0, 0.0]
            pose_giro = [-2.5, 0.3, 0.0, 0.5, 0.0, -0.0, 0.4]
            
            for _ in range(2):
                send_walk_cmd('a') 
                self.move_to_pose(pose_arriba, duration=1.1)
                self.wait_until_reached()
                
                send_walk_cmd('d')
                self.move_to_pose(pose_abajo, duration=1.1)
                self.wait_until_reached()

            send_walk_cmd('stop')
            time.sleep(0.5)
            self.move_to_pose(pose_up, duration=1.0)
            self.wait_until_reached()
            self.move_to_pose(pose_next, duration=1.5)
            self.wait_until_reached()

            send_walk_cmd('q') 
            self.move_to_pose(pose_giro, duration=1.8)
            self.wait_until_reached()
            
            send_walk_cmd('stop')
            time.sleep(0.5)
            send_walk_cmd('e') 
            self.move_to_pose(pose_giro, duration=1.8)
            self.wait_until_reached()
            
            send_walk_cmd('stop')
            self.move_to_pose(pose_next, duration=1.5)
            self.wait_until_reached()
            self.move_to_pose(pose_up, duration=1.5)
            self.wait_until_reached()

            self.move_to_home(duration=1.5)
            self.wait_until_reached()
            self.release_control()

    def udp_listener_loop(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        try:
            server_socket.bind(('0.0.0.0', 5005))
            print("[INFO] 🎧 Escoltant emocions per xarxa en el port 5005...")
        except Exception as e:
            print(f"[ERROR] No s'ha pogut fer bind al port 5005: {e}")
            return
            
        while True:
            try:
                data, addr = server_socket.recvfrom(1024)
                cmd = data.decode('utf-8').strip().upper()
                if cmd in ["HAPPY", "NEUTRAL", "FRUSTRATED", "SAD", "ANGRY", "DANCE"]:
                    print(f"📥 Emoció rebuda des de {addr}: {cmd}")
                    self.play_emotion(cmd)
            except Exception as e:
                print(f"Error rebent comanda: {e}")

    def control_loop(self):
        if not self.state_received or self.model is None: return

        if not self.active_ik:
            return

        if self.trajectory_q:
            self.q_math = self.trajectory_q.pop(0)

        cmd_msg = LowCmd()
        
        # 1. DEIXEM LA MAJORIA DELS MOTORS EN MODE 0
        for j in range(29):
            cmd_msg.motor_cmd[j].mode = 0  
            cmd_msg.motor_cmd[j].q = 0.0
            cmd_msg.motor_cmd[j].dq = 0.0
            cmd_msg.motor_cmd[j].tau = 0.0
            cmd_msg.motor_cmd[j].kp = 0.0
            cmd_msg.motor_cmd[j].kd = 0.0

        # 1.5 FIXEM LA CINTURA PERQUÈ NO ES DESPLOME
        # /arm_sdk captura la cintura. Si la deixem a mode 0, cau fluixa.
        # L'hem de subjectar fort a la posició recta que tenia a l'inici.
        for i, waist_idx in enumerate(self.g1_waist):
            cmd_msg.motor_cmd[waist_idx].mode = 1
            cmd_msg.motor_cmd[waist_idx].q = self.home_waist_jpos[i]
            cmd_msg.motor_cmd[waist_idx].dq = 0.0
            cmd_msg.motor_cmd[waist_idx].tau = 0.0
            cmd_msg.motor_cmd[waist_idx].kp = self.kp * 1.5 # Extra de força per subjectar
            cmd_msg.motor_cmd[waist_idx].kd = self.kd * 1.5

        # 2. Llegim els angles per als braços
        comandos_brazos = {}
        for q_idx, g1_idx in self.pin_to_g1_q.items():
            ang = float(self.q_math[q_idx])
            if g1_idx in self.joint_safety_limits:
                ang = max(self.joint_safety_limits[g1_idx][0], min(self.joint_safety_limits[g1_idx][1], ang))
            comandos_brazos[g1_idx] = ang
            
        # 3. ACTIVEM (MODE 1) ELS 14 MOTORS DELS BRAÇOS
        for i in range(7):
            left_motor_id = self.g1_arm_left[i]
            right_motor_id = self.g1_arm_right[i]
            left_angle = comandos_brazos[left_motor_id]
            
            if i in [1, 2, 4, 6]:
                right_angle = -left_angle
            else:
                right_angle = left_angle

            cmd_msg.motor_cmd[left_motor_id].mode = 1
            cmd_msg.motor_cmd[left_motor_id].q = left_angle
            cmd_msg.motor_cmd[left_motor_id].kp = self.kp
            cmd_msg.motor_cmd[left_motor_id].kd = self.kd

            cmd_msg.motor_cmd[right_motor_id].mode = 1
            cmd_msg.motor_cmd[right_motor_id].q = right_angle
            cmd_msg.motor_cmd[right_motor_id].kp = self.kp
            cmd_msg.motor_cmd[right_motor_id].kd = self.kd

        # Avisem a l'SDK
        cmd_msg.motor_cmd[self.NOT_USED_JOINT].q = 1.0
        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = G1PhysicalEmotions()
    
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    
    try:
        while not node.state_received:
            time.sleep(0.1)
            
        print("\n" + "="*50)
        print("🤖 CONTROL D'EMOCIONS FÍSIC INICIAT 🤖")
        print("="*50)
        
        while True:
            cmd = input("\n> Introdueix emoció (HAPPY, NEUTRAL, FRUSTRATED, SAD, ANGRY, DANCE) o 'q' per eixir: ").strip().upper()
            if cmd == 'Q':
                break
            elif cmd in ["HAPPY", "NEUTRAL", "FRUSTRATED", "SAD", "ANGRY", "DANCE"]:
                node.play_emotion(cmd)
            else:
                print("[ERROR] Comanda no reconeguda. Torna-ho a intentar.")
                
    except KeyboardInterrupt:
        pass
    finally:
        node.release_control()
        time.sleep(0.5)
        rclpy.shutdown()
        os._exit(0)

if __name__ == '__main__':
    main()
