import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from unitree_hg.msg import LowCmd, LowState
import time
import threading
import numpy as np
import os
import socket
import sys
import csv
import pandas as pd
import h5py

    def load_recording(self, emotion):
        """Lee el archivo .parquet o .h5 y extrae solo la cintura y los brazos, ignorando manos y piernas."""
        frames = []
        
        # Soportamos ambos formatos por si tus compañeros te pasan uno u otro
        filepath_parquet = os.path.join(self.recordings_dir, f"{emotion}.parquet")
        filepath_h5 = os.path.join(self.recordings_dir, f"{emotion}.h5")
        
        raw_data = []
        
        try:
            if os.path.exists(filepath_parquet):
                df = pd.read_parquet(filepath_parquet)
                # En los datasets de NVIDIA/LeRobot, el estado suele ser una columna con listas
                # o múltiples columnas. Adaptamos para la lista anidada:
                if 'observation.state' in df.columns:
                    raw_data = df['observation.state'].tolist()
                    
            elif os.path.exists(filepath_h5):
                with h5py.File(filepath_h5, 'r') as f:
                    # En archivos HDF5 de teleoperación (Robomimic), el estado suele estar aquí
                    raw_data = f['obs']['state'][:]
            else:
                print(f"[ERROR] No se encontró {emotion}.parquet ni {emotion}.h5 en {self.recordings_dir}")
                return frames
                
            # --- LA CIRUGÍA DE ÍNDICES ---
            for row in raw_data:
                # Recortamos el array de 43 valores usando la información del modality.json
                waist = row[12:15]       # Cintura (3)
                left_arm = row[15:22]    # Brazo izquierdo (7)
                right_arm = row[29:36]   # Brazo derecho (7) -> ¡Saltamos los índices 22-28 que son la mano!
                
                # Juntamos los 17 motores en el orden que espera nuestro control_loop
                frame_17_motors = list(waist) + list(left_arm) + list(right_arm)
                frames.append(frame_17_motors)
                
        except Exception as e:
            print(f"[ERROR] Fallo al decodificar la grabación de {emotion}: {e}")
            
        return frames

# Inicializamos el canal de Unitree para los LEDs
sdk_path = "/root/unitree_sdk2_python"
if sdk_path not in sys.path: sys.path.append(sdk_path)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.vui.vui_client import VuiClient
ChannelFactoryInitialize(0)

def send_walk_cmd(cmd):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('127.0.0.1', 6000))
        s.sendall(cmd.encode('utf-8'))
        s.recv(1024)
        s.close()
    except: pass

class G1PlaybackEmotions(Node):
    def __init__(self):
        super().__init__('g1_playback_emotions')
        
        self.vui = VuiClient(); self.vui.Init()
        
        self.NOT_USED_JOINT = 29 
        self.kp = 60.0
        self.kd = 1.5
        self.dt = 0.02  # 50Hz (Las grabaciones deben estar a 50 FPS)
        
        self.cmd_pub = self.create_publisher(LowCmd, '/arm_sdk', 10)
        self.state_sub = self.create_subscription(LowState, '/lowstate', self.state_callback, qos_profile_sensor_data)
        
        self.state_received = False
        
        # Índices exactos de los motores del Unitree G1
        self.g1_waist = [12, 13, 14]
        self.g1_arm_left = [15, 16, 17, 18, 19, 20, 21]
        self.g1_arm_right = [22, 23, 24, 25, 26, 27, 28] 
        self.controlled_joints = self.g1_waist + self.g1_arm_left + self.g1_arm_right # 17 motores
        
        self.current_jpos = [0.0] * 29 
        self.home_jpos = [0.0] * 17 # Guardará la postura base de la que parte el robot
        
        self.active_playback = False       
        self.playback_frames = [] # Aquí se cargarán las filas del archivo CSV
        
        self.recordings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        
        self.timer = self.create_timer(self.dt, self.control_loop)
        print("[INFO] Esperando estado del robot físico...")

    def state_callback(self, msg):
        for i in range(29):
            if i < len(msg.motor_state):
                self.current_jpos[i] = msg.motor_state[i].q
                
        if not self.state_received:
            self.state_received = True
            
            # Capturar la pose natural del robot al arrancar como su "Home"
            for i, motor_idx in enumerate(self.controlled_joints):
                self.home_jpos[i] = self.current_jpos[motor_idx]
                
            print("[INFO] Robot conectado. Reproductor de emociones listo.")
            threading.Thread(target=self.udp_listener_loop, daemon=True).start()

    def change_led(self, r, g, b):
        try: self.vui.SetLedColor(r, g, b)
        except: pass

    def load_csv(self, emotion):
        """Lee el CSV y devuelve una lista de frames (cada frame es una lista de 17 radianes)"""
        filepath = os.path.join(self.recordings_dir, f"{emotion}.csv")
        frames = []
        if not os.path.exists(filepath):
            print(f"[ERROR] No se encontró el archivo {filepath}")
            return frames
            
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Convertimos las columnas de texto a float
                frame = [float(val) for val in row]
                frames.append(frame)
        return frames

    def transition_to(self, target_jpos, duration=1.0):
        """Crea una interpolación suave desde la pose actual hasta el target (para empezar o terminar grabaciones sin tirones)"""
        steps = int(duration / self.dt)
        start_jpos = [self.current_jpos[idx] for idx in self.controlled_joints]
        
        transition_frames = []
        for i in range(1, steps + 1):
            alpha = i / steps
            frame = [start_jpos[j] * (1 - alpha) + target_jpos[j] * alpha for j in range(len(self.controlled_joints))]
            transition_frames.append(frame)
            
        self.playback_frames.extend(transition_frames)
        self.active_playback = True
        
        # Esperamos a que acabe la transición
        while self.playback_frames: time.sleep(0.02)

    def play_emotion(self, emotion):
        print(f"\n▶️ Reproduciendo grabación de: {emotion}")
        
        # 1. Cargamos el archivo de la memoria de los compañeros
        frames = self.load_recording(emotion)
        if not frames: return
        
        # 2. Luces y comportamientos específicos
        if emotion == "HAPPY": self.change_led(0, 255, 0)
        elif emotion == "SAD": self.change_led(0, 0, 50)
        elif emotion == "ANGRY": 
            self.change_led(255, 0, 0)
            send_walk_cmd('w'); time.sleep(0.5); send_walk_cmd('stop')
        elif emotion == "DANCE": self.change_led(255, 0, 255)
        else: self.change_led(255, 255, 255) # Blanco por defecto

        self.active_playback = True

        # 3. Transición suave desde donde esté el robot AHORA, hasta el PRIMER FRAME del vídeo
        self.transition_to(frames[0], duration=0.8)
        
        # 4. Inyectamos la grabación completa al bucle de control
        self.playback_frames.extend(frames)
        
        # Esperamos a que la grabación se reproduzca entera
        while self.playback_frames: time.sleep(0.02)
        
        # 5. Volvemos suavemente a la postura base (Home)
        self.transition_to(self.home_jpos, duration=1.0)
        
        # 6. Liberamos motores
        self.active_playback = False
        cmd = LowCmd()
        for j in range(29): cmd.motor_cmd[j].mode = 0
        cmd.motor_cmd[self.NOT_USED_JOINT].q = 0.0
        self.cmd_pub.publish(cmd)
        
        self.change_led(0, 0, 255) # Restaurar azul
        print(f"⏹ Grabación de {emotion} finalizada.")

    def udp_listener_loop(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
        server_socket.bind(('0.0.0.0', 5005))
        while True:
            try:
                data, _ = server_socket.recvfrom(1024)
                cmd = data.decode('utf-8').strip().upper()
                if cmd in ["HAPPY", "NEUTRAL", "FRUSTRATED", "SAD", "ANGRY", "DANCE"]:
                    self.play_emotion(cmd)
            except: pass

    def control_loop(self):
        if not self.state_received: return
        if not self.active_playback or not self.playback_frames: return

        # Sacamos el siguiente frame de la grabación (una lista de 17 números)
        current_frame = self.playback_frames.pop(0)

        cmd_msg = LowCmd()
        
        # Todos los motores a 0 por defecto (piernas libres)
        for j in range(29):
            cmd_msg.motor_cmd[j].mode = 0  
            cmd_msg.motor_cmd[j].q = 0.0
            cmd_msg.motor_cmd[j].dq = 0.0
            cmd_msg.motor_cmd[j].tau = 0.0
            cmd_msg.motor_cmd[j].kp = 0.0
            cmd_msg.motor_cmd[j].kd = 0.0

        # Aplicamos la grabación a Cintura y Brazos (Modo 1 = Activo)
        for idx, motor_id in enumerate(self.controlled_joints):
            cmd_msg.motor_cmd[motor_id].mode = 1
            cmd_msg.motor_cmd[motor_id].q = current_frame[idx]
            cmd_msg.motor_cmd[motor_id].dq = 0.0
            cmd_msg.motor_cmd[motor_id].tau = 0.0
            cmd_msg.motor_cmd[motor_id].kp = self.kp
            cmd_msg.motor_cmd[motor_id].kd = self.kd

        # Avisamos al SDK que enviamos comandos
        cmd_msg.motor_cmd[self.NOT_USED_JOINT].q = 1.0
        self.cmd_pub.publish(cmd_msg)

def main(args=None):
    rclpy.init(args=args)
    node = G1PlaybackEmotions()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        node.active_playback = False
        time.sleep(0.5)
        rclpy.shutdown()

if __name__ == '__main__':
    main()
