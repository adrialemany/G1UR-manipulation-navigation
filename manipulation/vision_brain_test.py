# 1. Dar prioridad a PyTorch/YOLO
from ultralytics import YOLO
import torch

# 2. Resto de librerías
import sys, os, time, zmq
import cv2
import numpy as np
import pyrealsense2 as rs

# SDK de Unitree para el color
sdk_path = "/root/unitree_sdk2_python"
if sdk_path not in sys.path: 
    sys.path.append(sdk_path)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient

# ==========================================
# PARÁMETROS ÓPTICOS DE LA CÁMARA (Depth 640x480)
# ==========================================
CX = 320.0
CY = 240.0
FOCAL_LENGTH = 460.0  # Distancia focal del láser infrarrojo de la D435i
# ==========================================

def main():
    print("[INFO] Iniciando Visión 3D Pura (Origen: Cámara)...")
    
    # --- 1. CONEXIÓN DE COLOR (Vía Unitree SDK) ---
    ChannelFactoryInitialize(0, "eth0")
    video = VideoClient()
    video.Init()

    # --- 2. CONEXIÓN DE PROFUNDIDAD (Vía Intel SDK - Solo Depth) ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        pipeline.start(config)
        print("[INFO] ¡Láser de profundidad conectado con éxito!")
    except RuntimeError as e:
        print(f"❌ ERROR de RealSense: {e}")
        return

    # --- 3. CARGA DE IA Y RED ---
    model = YOLO('best.pt')

    context = zmq.Context()
    zmq_pub_sdk = context.socket(zmq.PUB)
    zmq_pub_sdk.bind("tcp://0.0.0.0:6001")

    print("✅ ¡Test de Extremos de Arista (Modo Cámara) ACTIVO!")
    print("👉 Abre tu script cliente en el portátil para ver el puerto 6001.")

    try:
        while True:
            ret, data = video.GetImageSample()
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            if ret == 0 and data and depth_frame:
                np_arr = np.frombuffer(bytes(data), np.uint8)
                color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if color_image is not None:
                    # IA busca la caja en la imagen a color
                    results = model.predict(color_image, conf=0.25, verbose=False)
                    annotated_frame = results[0].plot()

                    if len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            x1_c, y1_c, x2_c, y2_c = map(int, box.xyxy[0])
                            
                            # Escalar de la resolución de Unitree a la del mapa de profundidad (640x480)
                            h_color, w_color = color_image.shape[:2]
                            escala_x = 640 / w_color
                            escala_y = 480 / h_color
                            
                            x1_d = int(x1_c * escala_x)
                            y1_d = int(y1_c * escala_y)
                            x2_d = int(x2_c * escala_x)
                            y2_d = int(y2_c * escala_y)
                            centro_x_d = (x1_d + x2_d) // 2
                            
                            # --- Escaneo vertical del láser para localizar el quiebre de la arista ---
                            z_arista = 999.0
                            y_arista_d = y1_d
                            
                            for fila in range(max(0, y1_d), min(479, y2_d)):
                                valores_z = []
                                for col in range(max(0, centro_x_d - 5), min(639, centro_x_d + 6)):
                                    dist = depth_frame.get_distance(col, fila)
                                    if 0.05 < dist < 4.0:
                                        valores_z.append(dist)
                                        
                                if valores_z:
                                    z_promedio = np.median(valores_z)
                                    if z_promedio < z_arista:
                                        z_arista = z_promedio
                                        y_arista_d = fila
                            
                            # Si encontramos la arista con el láser, medimos extremos
                            if z_arista != 999.0:
                                # 📐 FÓRMULA DE PROYECCIÓN NATIVA (Origen: Centro Óptico de la Cámara)
                                # X es desvío lateral, Y desvío vertical, Z profundidad en línea recta
                                x_cam_izq = ((x1_d - CX) * z_arista) / FOCAL_LENGTH
                                x_cam_der = ((x2_d - CX) * z_arista) / FOCAL_LENGTH
                                y_cam_arista = ((y_arista_d - CY) * z_arista) / FOCAL_LENGTH
                                z_cam = z_arista
                                
                                # Dibujar HUD visual en el frame
                                y_arista_c = int(y_arista_d / escala_y)
                                cv2.line(annotated_frame, (x1_c, y_arista_c), (x2_c, y_arista_c), (0, 255, 255), 2)
                                cv2.circle(annotated_frame, (x1_c, y_arista_c), 6, (255, 0, 255), -1) # Extremo Izq
                                cv2.circle(annotated_frame, (x2_c, y_arista_c), 6, (0, 165, 255), -1) # Extremo Der
                                
                                # Generar textos informativos del espacio óptico
                                txt_izq = f"Izq Cam-> X:{x_cam_izq:.2f} Y:{y_cam_arista:.2f} Z:{z_cam:.2f}m"
                                txt_der = f"Der Cam-> X:{x_cam_der:.2f} Y:{y_cam_arista:.2f} Z:{z_cam:.2f}m"
                                
                                cv2.putText(annotated_frame, txt_izq, (x1_c, max(20, y_arista_c - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
                                cv2.putText(annotated_frame, txt_der, (x1_c, max(40, y_arista_c - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
                                
                                print(f"📸 [CAM FREQUENCY] Extremo IZQ -> X:{x_cam_izq:.2f}m | Y:{y_cam_arista:.2f}m | Z:{z_cam:.2f}m")
                                print(f"📸 [CAM FREQUENCY] Extremo DER -> X:{x_cam_der:.2f}m | Y:{y_cam_arista:.2f}m | Z:{z_cam:.2f}m")
                                print("-" * 50)

                    # Enviar fotograma procesado por ZMQ al portátil
                    _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    zmq_pub_sdk.send(buffer.tobytes())
                    
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[*] Apagando sistema visual...")
    finally:
        pipeline.stop()
        zmq_pub_sdk.close()
        context.term()
        print("[*] Cámara RealSense liberada correctamente.")

if __name__ == '__main__':
    main()
