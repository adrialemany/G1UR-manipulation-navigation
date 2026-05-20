import cv2
import os
import sys
import re
import time
import threading
import numpy as np

# --- IMPORTACIONES SDK UNITREE ---
sdk_path = "/root/unitree_sdk2_python"
if sdk_path not in sys.path: sys.path.append(sdk_path)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

# Variables globales para el hilo
latest_frame = None
running = True

def video_loop(video_client):
    """Hilo para consumir constantemente los frames del SDK y evitar delay/backlog"""
    global latest_frame, running
    while running:
        # GetImageSample devuelve code=0 si es exitoso, y data con los bytes de la imagen
        code, data = video_client.GetImageSample()
        if code == 0 and data:
            # Convertir los bytes directamente a un frame de OpenCV
            buffer = np.frombuffer(bytes(data), dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
            if frame is not None:
                latest_frame = frame
                
        time.sleep(0.03)

def main():
    global latest_frame, running
    
    # 1. Crear carpeta dataset automáticamente
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"[INFO] Carpeta 'dataset' creada en: {dataset_dir}")

    # 2. Buscar por qué número de foto vamos
    max_num = 0
    if os.path.exists(dataset_dir):
        for f in os.listdir(dataset_dir):
            match = re.match(r"caja_(\d+)\.jpg", f)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num

    next_num = max_num + 1

    # 3. Inicializar Unitree SDK
    print("[INFO] Inicializando red Unitree...")
    # Usamos "eth0" tal como lo tenías configurado en tu g1_server.py
    ChannelFactoryInitialize(0, "eth0") 
    
    print("[INFO] Conectando al hardware de vídeo de Unitree...")
    video = VideoClient()
    video.SetTimeout(3.0)
    video.Init()

    # 4. Arrancar el hilo que mantendrá la imagen actualizada sin delay
    threading.Thread(target=video_loop, args=(video,), daemon=True).start()

    print("[INFO] Esperando a recibir el primer frame...")
    while latest_frame is None and running:
        time.sleep(0.1)

    print("\n" + "="*50)
    print("📸 RECOLECTOR DE DATASET (UNITREE SDK_CAM) 📸")
    print("="*50)
    print(f"Empezando a guardar a partir de: caja_{next_num}.jpg")
    
    try:
        while True:
            comando = input("\n> Pulsa [ENTER] para hacer foto, o escribe 'q' para salir: ").strip().lower()
            
            if comando == 'q':
                break
                
            # Si le damos a enter y tenemos una imagen reciente...
            if latest_frame is not None:
                # Hacemos una copia rápida en memoria por si el hilo la actualiza justo ahora
                frame_to_save = latest_frame.copy() 
                
                filename = os.path.join(dataset_dir, f"caja_{next_num}.jpg")
                cv2.imwrite(filename, frame_to_save)
                
                print(f"✅ ¡Foto guardada! -> {filename}")
                next_num += 1
            else:
                print("❌ [ERROR] No hay imagen disponible de la cámara del robot.")
                
    except KeyboardInterrupt:
        print("\n[INFO] Cancelado por teclado...")
    finally:
        running = False
        print("[INFO] Fin del programa. ¡A recopilar!")

if __name__ == "__main__":
    main()
