import os
import sys
import time
import math
import numpy as np
import cv2
import zmq
import multiprocessing as mp

# --- IA y Visión ---
from ultralytics import YOLO
import pyrealsense2 as rs

# --- ROS2 y Cinemática ---
import rclpy
from rclpy.node import Node
from unitree_hg.msg import LowCmd, LowState
import pinocchio as pin
from std_msgs.msg import Float32MultiArray

# --- Unitree SDK ---
sdk_path = "/root/unitree_sdk2_python"
if sdk_path not in sys.path: 
    sys.path.append(sdk_path)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

# ==========================================
# ⚙️ REGLAS DEL JUEGO Y DIMENSIONES
# ==========================================
LONGITUD_MANO = 0.16        
MARGEN_APROXIMACION = 0.15  
PENETRACION_AGARRE = 0.03   
DISTANCIA_ALCANZABLE = 0.55 

CX, CY = 320.0, 240.0
FOCAL_LENGTH = 460.0

# ==========================================
# 📐 TRANSFORMADOR CAD
# ==========================================
def transform_camera_to_pelvis(x_cam, y_cam, z_cam):
    OFFSET_X, OFFSET_Y, OFFSET_Z = 0.047645, 0.0, 0.462681
    pitch_rad = math.radians(48.0)
    x_rot = (z_cam * math.cos(pitch_rad)) + (y_cam * math.sin(pitch_rad))
    y_rot = -x_cam
    z_rot = (-z_cam * math.sin(pitch_rad)) - (y_cam * math.cos(pitch_rad))
    return np.array([x_rot + OFFSET_X, y_rot + OFFSET_Y, z_rot + OFFSET_Z])

# ==========================================
# 👁️ PROCESO INDEPENDIENTE DE VISIÓN (No ROS2)
# ==========================================
def vision_process(queue_out):
    """
    Este proceso corre aislado. Arranca el SDK de Unitree, la RealSense y YOLO.
    Cuando detecta una caja estática por 7s, envía las coordenadas por la cola
    al proceso principal de ROS2.
    """
    print("[Visión] Iniciando sistema óptico y SDK de Unitree...")
    ChannelFactoryInitialize(0, "eth0")
    video = VideoClient()
    video.Init()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        pipeline.start(config)
        print("[Visión] Láser 3D Listo.")
    except RuntimeError:
        print("❌ [Visión] Cámara ocupada. Mata videohub_ si falla.")
        return

    model = YOLO('best.pt')
    context = zmq.Context()
    zmq_pub_sdk = context.socket(zmq.PUB)
    zmq_pub_sdk.bind("tcp://0.0.0.0:6001")

    historial_caja = []
    caja_estable_desde = None
    estado_vision = "BUSCANDO"

    try:
        while True:
            ret, data = video.GetImageSample()
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            if ret == 0 and data and depth_frame:
                np_arr = np.frombuffer(bytes(data), np.uint8)
                color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if color_image is not None:
                    results = model.predict(color_image, conf=0.25, verbose=False)
                    annotated_frame = results[0].plot()
                    caja_detectada = False

                    if len(results[0].boxes) > 0:
                        box = results[0].boxes[0]
                        x1_c, y1_c, x2_c, y2_c = map(int, box.xyxy[0])
                        
                        escala_x = 640 / color_image.shape[1]
                        escala_y = 480 / color_image.shape[0]
                        x1_d, y1_d = int(x1_c * escala_x), int(y1_c * escala_y)
                        x2_d, y2_d = int(x2_c * escala_x), int(y2_c * escala_y)
                        cx_d = (x1_d + x2_d) // 2

                        z_arista, y_arista_d = 999.0, y1_d
                        for fila in range(max(0, y1_d), min(479, y2_d)):
                            vals = [depth_frame.get_distance(c, fila) for c in range(max(0, cx_d - 5), min(639, cx_d + 6))]
                            vals = [v for v in vals if 0.05 < v < 4.0]
                            if vals:
                                z_prom = np.median(vals)
                                if z_prom < z_arista:
                                    z_arista = z_prom
                                    y_arista_d = fila

                        if z_arista != 999.0:
                            caja_detectada = True
                            x_cam_izq = ((x1_d - CX) * z_arista) / FOCAL_LENGTH
                            x_cam_der = ((x2_d - CX) * z_arista) / FOCAL_LENGTH
                            y_cam_ar = ((y_arista_d - CY) * z_arista) / FOCAL_LENGTH
                            
                            p_izq_pelvis = transform_camera_to_pelvis(x_cam_izq, y_cam_ar, z_arista)
                            p_der_pelvis = transform_camera_to_pelvis(x_cam_der, y_cam_ar, z_arista)

                            centro_actual = (p_izq_pelvis + p_der_pelvis) / 2.0
                            historial_caja.append(centro_actual)
                            if len(historial_caja) > 30: historial_caja.pop(0)

                            movimiento = 0.0
                            if len(historial_caja) > 10:
                                hist = np.array(historial_caja)
                                movimiento = np.max(np.linalg.norm(hist - hist[-1], axis=1))

                            if movimiento < 0.03: 
                                if caja_estable_desde is None:
                                    caja_estable_desde = time.time()
                                
                                seg_estable = time.time() - caja_estable_desde
                                cv2.putText(annotated_frame, f"Fijando... {seg_estable:.1f}s/7s", (x1_c, max(30, y1_c - 15)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                                if seg_estable >= 7.0 and estado_vision == "BUSCANDO":
                                    # ¡Enviar orden a ROS2!
                                    coords = np.concatenate([p_izq_pelvis, p_der_pelvis])
                                    queue_out.put(coords)
                                    estado_vision = "AGARRANDO"
                            else:
                                caja_estable_desde = None

                            y_arista_c = int(y_arista_d / escala_y)
                            cv2.line(annotated_frame, (x1_c, y_arista_c), (x2_c, y_arista_c), (0, 255, 255), 2)
                            cv2.circle(annotated_frame, (x1_c, y_arista_c), 5, (255, 0, 255), -1)
                            cv2.circle(annotated_frame, (x2_c, y_arista_c), 5, (0, 165, 255), -1)

                    if not caja_detectada:
                        caja_estable_desde = None

                    cv2.putText(annotated_frame, f"Vision: {estado_vision}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    zmq_pub_sdk.send(buffer.tobytes())
                    
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        zmq_pub_sdk.close()
        context.term()


# ==========================================
# 🤖 NODO MAESTRO ROS2 (Control e IK)
# ==========================================
class G1MasterController(Node):
    def __init__(self, vision_queue):
        super().__init__('g1_master_controller')
        self.get_logger().info("Iniciando Cerebro Maestro (ROS2 + IK)")
        self.vision_queue = vision_queue
        
        self.NOT_USED_JOINT = 29 
        self.kp, self.kd = 60.0, 1.5
        self.dt = 0.02 
        
        self.cmd_pub = self.create_publisher(LowCmd, '/arm_sdk', 10)
        self.state_sub = self.create_subscription(LowState, '/lowstate', self.state_callback, 10)
        
        self.state_received = False
        self.current_jpos = [0.0] * 29 
        self.g1_arm_left = [15, 16, 17, 18, 19, 20, 21]
        self.g1_arm_right = [22, 23, 24, 25, 26, 27, 28]

        self.estado_robot = "ESPERANDO_VISION"
        self.tiempo_estado = time.time()
        self.coordenadas_agarre = {"izq": None, "der": None}
        
        self.active_ik = False
        self.use_6d = False
        self.target_l, self.target_r = None, None
        self.target_rot_l, self.target_rot_r = None, None
        self.hand_l_actual = np.zeros(3)
        self.hand_r_actual = np.zeros(3)
        self.init_kinematics()

        self.timer = self.create_timer(self.dt, self.control_loop)

    def init_kinematics(self):
        urdf_path = os.path.expanduser("~/robot_ws/src/g1pilot/description_files/urdf/g1_29dof.urdf")
        try:
            self.full_model = pin.buildModelFromUrdf(urdf_path)
            joints_to_lock = [
                "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
                "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            ]
            locked_ids = [self.full_model.getJointId(j) for j in joints_to_lock if self.full_model.existJointName(j)]
            q_neutral = pin.neutral(self.full_model)
            self.model = pin.buildReducedModel(self.full_model, locked_ids, q_neutral)
            self.data = self.model.createData()
            self.left_hand_id = self.model.getFrameId("left_rubber_hand")
            self.right_hand_id = self.model.getFrameId("right_rubber_hand")
            self.q_math = pin.neutral(self.model)
        except Exception as e:
            self.get_logger().error(f"Fallo cargando URDF: {e}")
            sys.exit(1)

        arm_names_l = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
        arm_names_r = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
        
        self.pin_to_g1_q = {}
        self.v_indices_l, self.v_indices_r = [], []
        
        for i, name in enumerate(arm_names_l):
            j_id = self.model.getJointId(name)
            self.pin_to_g1_q[self.model.joints[j_id].idx_q] = self.g1_arm_left[i]
            self.v_indices_l.append(self.model.joints[j_id].idx_v)
            
        for i, name in enumerate(arm_names_r):
            j_id = self.model.getJointId(name)
            self.pin_to_g1_q[self.model.joints[j_id].idx_q] = self.g1_arm_right[i]
            self.v_indices_r.append(self.model.joints[j_id].idx_v)

    def state_callback(self, msg):
        for i in range(29):
            if i < len(msg.motor_state):
                self.current_jpos[i] = msg.motor_state[i].q
        if not self.state_received:
            self.state_received = True

    def procesar_cola_vision(self):
        if not self.vision_queue.empty():
            coords = self.vision_queue.get()
            self.coordenadas_agarre["izq"] = coords[0:3]
            self.coordenadas_agarre["der"] = coords[3:6]
            self.iniciar_agarre()

    def iniciar_agarre(self):
        izq = self.coordenadas_agarre["izq"]
        if izq[0] > DISTANCIA_ALCANZABLE:
            self.get_logger().warn(f"Caja inalcanzable (X={izq[0]:.2f}m).")
            self.estado_robot = "ESPERANDO_VISION"
            return
            
        self.get_logger().info("🔥 ¡A POR LA CAJA! Activando Cinemática Bimanual...")
        self.active_ik = True
        
        R_pitch = pin.utils.rpyToMatrix(0, 0, 0)
        R_roll_l = pin.utils.rpyToMatrix(math.radians(-80), 0, 0) 
        R_roll_r = pin.utils.rpyToMatrix(math.radians(80), 0, 0)
        
        self.target_rot_l = R_roll_l @ R_pitch
        self.target_rot_r = R_roll_r @ R_pitch
        self.use_6d = True

        self.estado_robot = "APROXIMACION_AMPLIA"
        self.tiempo_estado = time.time()

    def maquina_de_estados(self):
        if self.estado_robot == "ESPERANDO_VISION": return
        now = time.time()
        
        izq = self.coordenadas_agarre["izq"]
        der = self.coordenadas_agarre["der"]
        
        target_x = izq[0] - LONGITUD_MANO
        target_z = izq[2] 
        
        if self.estado_robot == "APROXIMACION_AMPLIA":
            self.target_l = np.array([target_x, izq[1] + MARGEN_APROXIMACION, target_z])
            self.target_r = np.array([target_x, der[1] - MARGEN_APROXIMACION, target_z])
            
            err_l = np.linalg.norm(self.target_l - self.hand_l_actual)
            err_r = np.linalg.norm(self.target_r - self.hand_r_actual)
            
            if (err_l < 0.04 and err_r < 0.04) or (now - self.tiempo_estado > 5.0):
                self.get_logger().info("Brazos listos. ¡Cerrando pinzas!")
                self.estado_robot = "CERRAR_AGARRE"
                self.tiempo_estado = now

        elif self.estado_robot == "CERRAR_AGARRE":
            self.target_l = np.array([target_x, izq[1] - PENETRACION_AGARRE, target_z])
            self.target_r = np.array([target_x, der[1] + PENETRACION_AGARRE, target_z])
            
            if now - self.tiempo_estado > 3.0: 
                self.get_logger().info("Caja agarrada. ¡Arriba!")
                self.estado_robot = "LEVANTAR"
                self.tiempo_estado = now
                
        elif self.estado_robot == "LEVANTAR":
            self.target_l[2] = target_z + 0.15
            self.target_r[2] = target_z + 0.15
            
            if now - self.tiempo_estado > 4.0:
                self.get_logger().info("✅ Agarre Completado.")
                self.estado_robot = "MISION_COMPLETADA"

    def compute_arm_ik(self, target_pos, hand_actual, frame_id, v_indices, target_rot):
        err_pos = target_pos - hand_actual
        err_norm = np.linalg.norm(err_pos)
        if err_norm > 0.02: err_pos = (err_pos / err_norm) * 0.02

        J = pin.computeFrameJacobian(self.model, self.data, self.q_math, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J_arm = J[:, v_indices]

        if self.use_6d and target_rot is not None:
            R_curr = self.data.oMf[frame_id].rotation
            R_err = target_rot @ R_curr.T
            theta = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
            w = np.zeros(3)
            if theta > 1e-5:
                w = (theta / (2 * np.sin(theta))) * np.array([R_err[2, 1] - R_err[1, 2], R_err[0, 2] - R_err[2, 0], R_err[1, 0] - R_err[0, 1]])
            err_6d = np.concatenate([err_pos, w])
            pseudo_inv = J_arm.T @ np.linalg.inv(J_arm @ J_arm.T + (0.05**2) * np.eye(6))
            return pseudo_inv @ err_6d * 2.0
        else:
            pseudo_inv = J_arm[:3, :].T @ np.linalg.inv(J_arm[:3, :] @ J_arm[:3, :].T + (0.05**2) * np.eye(3))
            return pseudo_inv @ err_pos * 2.0

    def control_loop(self):
        if not self.state_received: return
        
        self.procesar_cola_vision()

        cmd = LowCmd()
        if not self.active_ik:
            cmd.motor_cmd[self.NOT_USED_JOINT].q = 0.0
            self.cmd_pub.publish(cmd)
            for q_idx, g1_idx in self.pin_to_g1_q.items():
                self.q_math[q_idx] = self.current_jpos[g1_idx]
            return

        self.maquina_de_estados()
        
        pin.forwardKinematics(self.model, self.data, self.q_math)
        pin.updateFramePlacements(self.model, self.data)
        self.hand_l_actual = self.data.oMf[self.left_hand_id].translation
        self.hand_r_actual = self.data.oMf[self.right_hand_id].translation

        dq = np.zeros(self.model.nv)
        if self.target_l is not None:
            dq_l = self.compute_arm_ik(self.target_l, self.hand_l_actual, self.left_hand_id, self.v_indices_l, self.target_rot_l)
            for i, v_idx in enumerate(self.v_indices_l): dq[v_idx] = dq_l[i]
            
        if self.target_r is not None:
            dq_r = self.compute_arm_ik(self.target_r, self.hand_r_actual, self.right_hand_id, self.v_indices_r, self.target_rot_r)
            for i, v_idx in enumerate(self.v_indices_r): dq[v_idx] = dq_r[i]

        self.q_math = pin.integrate(self.model, self.q_math, dq * self.dt)

        for q_idx, g1_idx in self.pin_to_g1_q.items():
            cmd.motor_cmd[g1_idx].q = float(self.q_math[q_idx])
            cmd.motor_cmd[g1_idx].dq = 0.0
            cmd.motor_cmd[g1_idx].tau = 0.0
            cmd.motor_cmd[g1_idx].kp = self.kp
            cmd.motor_cmd[g1_idx].kd = self.kd

        cmd.motor_cmd[self.NOT_USED_JOINT].q = 1.0
        self.cmd_pub.publish(cmd)

    def release(self):
        cmd = LowCmd()
        cmd.motor_cmd[self.NOT_USED_JOINT].q = 0.0
        self.cmd_pub.publish(cmd)


def main(args=None):
    # Crear cola de comunicación entre procesos
    vision_queue = mp.Queue()
    
    # Arrancar el proceso de Visión aislado
    v_proc = mp.Process(target=vision_process, args=(vision_queue,))
    v_proc.start()

    # Arrancar ROS2 en el proceso principal
    rclpy.init(args=args)
    node = G1MasterController(vision_queue)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.release()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        v_proc.terminate()
        v_proc.join()

if __name__ == '__main__':
    # Necesario en Ubuntu para evitar que ROS2 herede contextos del sistema
    mp.set_start_method('spawn')
    main()
