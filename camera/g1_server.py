import sys, os, time, socket, threading, zmq, subprocess
import cv2
import numpy as np
import sounddevice as sd
from gtts import gTTS

# Configuración de Audio
AUDIO_DEVICE = "hw:0,0"
CHANNELS = 1
RATE = 16000  
CHUNK = 1024

sdk_path = "/root/unitree_sdk2_python"
if sdk_path not in sys.path: sys.path.append(sdk_path)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.go2.video.video_client import VideoClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

# Directorios para recursos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
JUMPSCARE_PATH = os.path.join(BASE_DIR, "Fnaf Jumpscare Sound Effect.mp3")

move_state = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0, "moving": False}
last_ping_time = time.time()

def locomotion_loop(loco):
    global last_ping_time
    while True:
        if time.time() - last_ping_time > 1.5:
            if move_state["moving"]:
                print("⚠️ WATCHDOG: Parada de emergencia.")
                move_state["moving"] = False
                loco.Move(0.0, 0.0, 0.0)
        if move_state["moving"]:
            loco.Move(move_state["vx"], move_state["vy"], move_state["vyaw"])
        time.sleep(0.05)

def usb_cam_loop(zmq_pub_usb):
    cap = cv2.VideoCapture(6)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            zmq_pub_usb.send(buffer.tobytes())
        time.sleep(0.03)

def audio_stream_loop(zmq_context):
    print(f"[*] Capturando audio de {AUDIO_DEVICE}...")
    pub_audio = zmq_context.socket(zmq.PUB)
    pub_audio.bind("tcp://0.0.0.0:6003")

    def callback(indata, frames, time_info, status):
        pub_audio.send(indata.tobytes())

    try:
        with sd.InputStream(device=AUDIO_DEVICE, channels=CHANNELS, samplerate=RATE, 
                            blocksize=CHUNK, dtype='float32', callback=callback):
            while True: time.sleep(1)
    except Exception as e:
        print(f"❌ Error en stream de audio: {e}")
        
def handle_client(conn, addr, loco, arm, audio_sdk):
    global last_ping_time
    with conn:
        try:
            data = conn.recv(1024)
            if not data: 
                return
            
            cmd = data.decode('utf-8').strip()
            last_ping_time = time.time()

            if cmd.startswith('speak:'):
                phrase = cmd.split(':', 1)[1].strip()
                if phrase.lower() == "jumpscare":
                    audio_sdk.SetVolume(50)
                    subprocess.run(["ffmpeg", "-y", "-i", JUMPSCARE_PATH, "-f", "s16le", "-ar", "16000", "-ac", "1", "temp.pcm"])
                    with open("temp.pcm", "rb") as f:
                        audio_sdk.PlayStream(f"j_{int(time.time())}", "1", f.read())
                else:
                    try:
                        tts = gTTS(text=phrase, lang='es')
                        tts.save(f"temp_tts_{addr[1]}.mp3") # Nombre único por puerto para evitar colisiones entre hilos
                        subprocess.run(["ffmpeg", "-y", "-i", f"temp_tts_{addr[1]}.mp3", "-f", "s16le", "-ar", "16000", "-ac", "1", f"temp_tts_{addr[1]}.pcm"], 
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        with open(f"temp_tts_{addr[1]}.pcm", "rb") as f:
                            audio_sdk.PlayStream(f"tts_{int(time.time())}", "1", f.read())
                    except Exception as e:
                        print(f"Error generant TTS en castellà: {e}")
                        audio_sdk.TtsMaker(phrase, 1)
            elif cmd == 'zero': loco.ZeroTorque()
            elif cmd == 'damp': loco.Damp()
            elif cmd == 'stand': loco.Squat2StandUp()
            elif cmd == 'squat': loco.StandUp2Squat()
            elif cmd == 'w': move_state.update({"vx": 0.4, "moving": True})
            elif cmd == 's': move_state.update({"vx": -0.4, "moving": True})
            elif cmd == 'stop': move_state["moving"] = False; loco.Move(0,0,0)
            elif cmd == 'ready': 
                print("DEBUG: Estado Ready (FSM 4: Lock Standing)")
                loco.SetFsmId(4) 
            elif cmd == 'motion': 
                print("DEBUG: Estado Motion (Start - Main Control)")
                loco.Start()
            elif cmd in ['0','1','2','3']:
                ids = {'0':99, '1':27, '2':26, '3':17}
                arm.ExecuteAction(ids[cmd])
            elif cmd == 'a': move_state.update({"vx": 0.0, "vy": 0.2, "vyaw": 0.0, "moving": True})
            elif cmd == 'd': move_state.update({"vx": 0.0, "vy": -0.2, "vyaw": 0.0, "moving": True})
            elif cmd == 'q': move_state.update({"vx": 0.0, "vy": 0.0, "vyaw": 0.5, "moving": True})
            elif cmd == 'e': move_state.update({"vx": 0.0, "vy": 0.0, "vyaw": -0.5, "moving": True})
                
            conn.sendall(b"FIN")
        except Exception as e: 
            print(f"Error CMD: {e}")

def main():
    global last_ping_time
    ChannelFactoryInitialize(0, "eth0")
    
    arm = G1ArmActionClient(); arm.Init()
    video = VideoClient(); video.Init()
    loco = LocoClient(); loco.Init()
    audio_sdk = AudioClient(); audio_sdk.Init()
    audio_sdk.tts_index = 1

    context = zmq.Context()

    # Hilos de sensores
    threading.Thread(target=locomotion_loop, args=(loco,), daemon=True).start()
    
    zmq_pub_sdk = context.socket(zmq.PUB)
    zmq_pub_sdk.bind("tcp://0.0.0.0:6001")
    def sdk_cam():
        while True:
            c, d = video.GetImageSample()
            if c == 0 and d: zmq_pub_sdk.send(bytes(d))
            time.sleep(0.03)
    threading.Thread(target=sdk_cam, daemon=True).start()

    zmq_pub_usb = context.socket(zmq.PUB)
    zmq_pub_usb.bind("tcp://0.0.0.0:6002")
    threading.Thread(target=usb_cam_loop, args=(zmq_pub_usb,), daemon=True).start()

    threading.Thread(target=audio_stream_loop, args=(context,), daemon=True).start()

    script2_path = os.path.join(SCRIPT_DIR, "emotions_g1.py")
    
    print(f"[*] Iniciando proceso de emociones: {script2_path}")
    # sys.executable asegura que use el mismo entorno de Python (útil si usas entornos virtuales de ROS2)
    process_emotions = subprocess.Popen([sys.executable, script2_path])
    # -------------------------------------------------------------

    try:
        # Servidor de Comandos TCP
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', 6000)); s.listen()
            print("[*] SERVIDOR G1 ONLINE (Puerto 6000-6003)")

            while True:
                conn, addr = s.accept()
                # Inicia un hilo por cada cliente que se conecta
                client_thread = threading.Thread(
                    target=handle_client, 
                    args=(conn, addr, loco, arm, audio_sdk),
                    daemon=True
                )
                client_thread.start()
                    
    except KeyboardInterrupt:
        print("\n[*] Deteniendo servidor principal...")
    finally:
        # --- NUEVO: LIMPIEZA DEL SUBPROCESO AL CERRAR ---
        print("[*] Matando el proceso de emociones dependiente...")
        process_emotions.terminate()
        process_emotions.wait()
        print("[*] Todo apagado correctamente.")

if __name__ == '__main__':
    main()
