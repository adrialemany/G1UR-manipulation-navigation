import sys, os, time, zmq
import cv2
import numpy as np
from ultralytics import YOLO

# 1. Configurar la ruta del SDK de Unitree
sdk_path = "/root/unitree_sdk2_python"
if sdk_path not in sys.path: 
    sys.path.append(sdk_path)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

def main():
    print("[INFO] Conectando con los nervios ópticos del Unitree G1...")
    # Inicializamos la red interna del robot
    ChannelFactoryInitialize(0, "eth0")
    
    video = VideoClient()
    video.Init()

    print("[INFO] Cargando el Cerebro YOLOv8 (best.pt)...")
    model = YOLO('best.pt')

    print("[INFO] Abriendo túnel ZMQ en el puerto 6001...")
    context = zmq.Context()
    zmq_pub_sdk = context.socket(zmq.PUB)
    zmq_pub_sdk.bind("tcp://0.0.0.0:6001")

    print("✅ ¡Test de Visión Robotizado ACTIVO!")
    print("👉 Abre el script cliente en tu portátil para ver el puerto 6001.")
    print("🛑 Presiona Ctrl+C para apagar este test.")

    try:
        while True:
            # 1. Obtener los bytes crudos de la cámara del robot
            ret, data = video.GetImageSample()
            
            if ret == 0 and data:
                # 2. Transformar esos bytes en una matriz de imagen para OpenCV
                np_arr = np.frombuffer(bytes(data), np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # 3. La IA busca la caja negra (Confianza adaptada a 0.25)
                    results = model.predict(frame, conf=0.25, verbose=False)
                    
                    # 4. Dibujamos el polígono en el frame
                    annotated_frame = results[0].plot()
                    
                    # 5. Volvemos a comprimir la imagen YA PINTADA y la enviamos por la red
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    zmq_pub_sdk.send(buffer.tobytes())
                else:
                    # Plan B: Si un frame llega roto desde la cámara, enviamos los bytes originales
                    zmq_pub_sdk.send(bytes(data))
                    
            # 6. Pequeño respiro para no saturar la CPU del robot
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[*] Apagando el test de visión...")
    finally:
        # Limpieza higiénica de los puertos al salir
        zmq_pub_sdk.close()
        context.term()
        print("[*] Puerto 6001 cerrado correctamente. Ya puedes volver a encender tu servidor principal.")

if __name__ == '__main__':
    main()
