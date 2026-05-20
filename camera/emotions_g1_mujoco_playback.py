import os
import sys
import time
import socket
import threading
import numpy as np
import json
import pandas as pd
import h5py

# Configuración DDS obligatoria para la simulación
os.environ["CYCLONEDDS_URI"] = """<CycloneDDS>
    <Domain>
        <SharedMemory>
            <Enable>false</Enable>
        </SharedMemory>
    </Domain>
</CycloneDDS>"""

sdk_path = "/root/unitree_sdk2_python"
if sdk_path not in sys.path: sys.path.append(sdk_path)

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

class G1MujocoPlaybackEmotions:
    def __init__(self):
        self.dt = 0.02  # 50Hz (Las grabaciones están a 50 FPS)
        
        # Socket UDP para enviar comandos a MuJoCo (run_sim_ai_g1.py)
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target_address = ('127.0.0.1', 9876)
        
        self.state_received = False
        self.tick_count = 0
        
        # Índices de los motores en el G1 Estándar (29 DoF)
        self.g1_waist = [12, 13, 14]
        self.g1_arm_left = [15, 16, 17, 18, 19, 20, 21]
        self.g1_arm_right = [22, 23, 24, 25, 26, 27, 28] 
        self.controlled_joints = self.g1_waist + self.g1_arm_left + self.g1_arm_right # 17 motores
        
        self.current_jpos = [0.0] * 29 
        self.home_jpos = [0.0] * 17 # Guardará la postura base (ready)
        
        self.active_playback = False       
        self.playback_frames = [] 
        
        self.recordings_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grabaciones")
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
            print(f"[AVISO] Se ha creado la carpeta '{self.recordings_dir}'. Pon ahí los .parquet")
        
        # Inicializar DDS
        ChannelFactoryInitialize(1, "lo") 
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.state_callback, 10)
        
        print("[INFO] Esperando estado de la simulación MuJoCo...")

    def state_callback(self, msg: LowState_):
        self.tick_count += 1
        for i in range(29):
            if i < len(msg.motor_state):
                self.current_jpos[i] = msg.motor_state[i].q
                
        # Esperamos unos 100 ticks para que el robot se ponga de pie y estabilice el "ready"
        if not self.state_received and self.tick_count > 100:
            self.state_received = True
            
            # Capturar la pose natural del robot al arrancar como su "Home"
            for i, motor_idx in enumerate(self.controlled_joints):
                self.home_jpos[i] = self.current_jpos[motor_idx]
                
            print("[INFO] Robot estabilizado. Reproductor de emociones listo.")
            threading.Thread(target=self.control_loop, daemon=True).start()
            threading.Thread(target=self.udp_listener_loop, daemon=True).start()

    def load_recording(self, emotion):
        """Lee el archivo .parquet o .h5 y recorta la cintura y los brazos, ignorando los dedos."""
        frames = []
        filepath_parquet = os.path.join(self.recordings_dir, f"{emotion}.parquet")
        filepath_h5 = os.path.join(self.recordings_dir, f"{emotion}.h5")
        
        raw_data = []
        try:
            if os.path.exists(filepath_parquet):
                df = pd.read_parquet(filepath_parquet)
                if 'observation.state' in df.columns:
                    raw_data = df['observation.state'].tolist()
            elif os.path.exists(filepath_h5):
                with h5py.File(filepath_h5, 'r') as f:
                    raw_data = f['obs']['state'][:]
            else:
                print(f"[ERROR] No se encontró {emotion}.parquet ni {emotion}.h5 en {self.recordings_dir}")
                return frames
                
            # --- CIRUGÍA DE ÍNDICES (De 43 DoF a 17 DoF) ---
            for row in raw_data:
                waist = row[12:15]       # Cintura (3)
                left_arm = row[15:22]    # Brazo izquierdo (7)
                right_arm = row[29:36]   # Brazo derecho (7) -> Saltamos los dedos (22-28)
                
                frame_17_motors = list(waist) + list(left_arm) + list(right_arm)
                frames.append(frame_17_motors)
                
        except Exception as e:
            print(f"[ERROR] Fallo al decodificar la grabación de {emotion}: {e}")
            
        return frames

    def transition_to(self, target_jpos, duration=1.0):
        """Interpolación suave desde la pose actual hasta el target."""
        steps = max(1, int(duration / self.dt))
        start_jpos = [self.current_jpos[idx] for idx in self.controlled_joints]
        
        transition_frames = []
        for i in range(1, steps + 1):
            alpha = i / steps
            frame = [start_jpos[j] * (1 - alpha) + target_jpos[j] * alpha for j in range(len(self.controlled_joints))]
            transition_frames.append(frame)
            
        self.playback_frames.extend(transition_frames)
        self.active_playback = True
        
        while self.playback_frames: time.sleep(0.02)

    def play_emotion(self, emotion):
        print(f"\n▶️ Reproduciendo grabación de: {emotion}")
        
        frames = self.load_recording(emotion)
        if not frames: return
        
        # Enviar comando de color al g1_server.py
        if emotion == "HAPPY": send_walk_cmd('led:0,255,0')
        elif emotion == "SAD": send_walk_cmd('led:0,0,50')
        elif emotion == "ANGRY": 
            send_walk_cmd('led:255,0,0')
            send_walk_cmd('w'); time.sleep(0.5); send_walk_cmd('stop')
        elif emotion == "DANCE": send_walk_cmd('led:255,0,255')
        else: send_walk_cmd('led:255,255,255')

        self.active_playback = True

        # Transición al primer frame del vídeo
        self.transition_to(frames[0], duration=0.8)
        
        # Reproducimos la teleoperación
        self.playback_frames.extend(frames)
        while self.playback_frames: time.sleep(0.02)
        
        # Volvemos a la postura base (Home)
        self.transition_to(self.home_jpos, duration=1.0)
        
        # Liberamos motores (En MuJoCo se hace vaciando la cola y mandando {})
        self.active_playback = False
        try:
            self.udp_sock.sendto(json.dumps({}).encode('utf-8'), self.target_address)
        except: pass
        
        send_walk_cmd('led:0,0,255') # Restaurar azul
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
        while True:
            t_start = time.time()
            if not self.state_received: 
                time.sleep(self.dt)
                continue

            # Si no hay reproducción, mandamos un JSON vacío para que el robot respire
            if not self.active_playback or not self.playback_frames:
                try:
                    self.udp_sock.sendto(json.dumps({}).encode('utf-8'), self.target_address)
                except: pass
                time.sleep(self.dt)
                continue

            # Sacamos el siguiente frame de la grabación (lista de 17 números)
            current_frame = self.playback_frames.pop(0)

            # Construimos el diccionario JSON: {"12": rads, "13": rads...}
            comandos_brazos = {}
            for idx, motor_id in enumerate(self.controlled_joints):
                comandos_brazos[str(motor_id)] = float(current_frame[idx])

            # Enviamos el comando a MuJoCo
            try:
                self.udp_sock.sendto(json.dumps(comandos_brazos).encode('utf-8'), self.target_address)
            except Exception:
                pass

            # Mantenemos los 50Hz clavados
            time.sleep(max(0.0, self.dt - (time.time() - t_start)))

if __name__ == '__main__':
    node = G1MujocoPlaybackEmotions()
    try:
        while not node.state_received:
            time.sleep(0.1)
            
        print("\n" + "="*50)
        print("🤖 REPRODUCTOR DE GRABACIONES EN MUJOCO INICIADO 🤖")
        print("="*50)
        
        while True:
            cmd = input("\n> Introdueix emoció (HAPPY, NEUTRAL, FRUSTRATED, SAD, ANGRY, DANCE) o 'q' per eixir: ").strip().upper()
            if cmd == 'Q':
                break
            elif cmd in ["HAPPY", "NEUTRAL", "FRUSTRATED", "SAD", "ANGRY", "DANCE"]:
                node.play_emotion(cmd)
            else:
                print("[ERROR] Comanda no reconeguda.")
                
    except KeyboardInterrupt:
        print("\nSaliendo...")
        os._exit(0)
