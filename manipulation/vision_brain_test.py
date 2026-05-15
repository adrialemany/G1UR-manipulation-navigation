import cv2
import numpy as np
import time
import threading

# Parámetros de la cámara (Ajustar según la resolución real que uses en el G1)
IMG_W, IMG_H = 640, 480
CX, CY = 320.0, 240.0
FOCAL_LENGTH = 460.0

class VisionBrainTester:
    def __init__(self):
        print("[INFO] Iniciando Corteza Visual (Modo Solo Lectura)...")
        
        # Buffers para las imágenes
        self.frame_rgb = None
        self.frame_depth = None
        self.running = True
        
        # --- HILO DE CAPTURA DE CÁMARA ---
        # TODO: Aquí debes conectar tu cámara física. 
        # Si usas ROS 2, aquí irían las suscripciones. 
        # Si usas la RealSense directa, aquí iría el pipeline.start()
        # He dejado un esqueleto genérico con OpenCV para la webcam por defecto a modo de prueba.
        self.cap = cv2.VideoCapture(0) # Cambiar por el ID de la cámara del robot si aplica
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
        
        threading.Thread(target=self.camera_loop, daemon=True).start()

    def camera_loop(self):
        """Hilo dedicado a obtener los frames lo más rápido posible"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_rgb = frame
                
                # SIMULACIÓN DE DEPTH PARA PRUEBAS (Borrar cuando conectes la cámara real)
                # Como una webcam normal no tiene depth, creamos una matriz falsa para que el código no pete.
                # ¡Sustituye esto por la lectura real de tu cámara de profundidad!
                if self.frame_depth is None:
                    self.frame_depth = np.ones((IMG_H, IMG_W), dtype=np.float32) * 1.5 
                    
            time.sleep(0.03)

    def get_robust_depth(self, x, y, w, h):
        """
        Escanea el centro de la caja. El plástico negro absorbe infrarrojos, 
        por lo que habrá muchos píxeles con valor 0.0 o NaN. Este método los ignora.
        """
        if self.frame_depth is None: return None

        # Cogemos el tercio central de la caja para evitar medir la mesa por error
        y_start = int(y + h * 0.3)
        y_end   = int(y + h * 0.7)
        x_start = int(x + w * 0.3)
        x_end   = int(x + w * 0.7)

        # Recortamos la región de interés (ROI)
        roi_depth = self.frame_depth[y_start:y_end, x_start:x_end]

        # Filtramos: ignoramos los menores a 5cm (ruido/agujeros) y mayores a 3m (lejos)
        valid_depths = roi_depth[(roi_depth > 0.05) & (roi_depth < 3.0)]

        if len(valid_depths) < 15:
            return None # La cámara no puede ver el negro, está ciega en esta zona

        # Usamos la mediana para ignorar picos de ruido
        return float(np.median(valid_depths))

    def detect_black_box(self, frame):
        """Procesa el frame RGB para buscar la caja de plástico negro"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 1. Rango para el color Negro / Gris oscuro
        # El negro puro tiene Value(Brillo)=0. Subimos hasta 60 para coger reflejos grises.
        # La saturación y el tono dan igual en el negro.
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60]) 
        
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # 2. Operaciones Morfológicas (MAGIA PARA PLÁSTICO BRILLANTE)
        # El brillo hace "agujeros" en la máscara. MORPH_CLOSE los rellena juntando las manchas.
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # MORPH_OPEN elimina el "ruido de sal" (sombras pequeñas, cables oscuros sueltos)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_box = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filtramos por área para ignorar manchas pequeñas
            if area > 1000: 
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Filtro por proporción: Una caja suele ser cuadrada o un poco rectangular (evitamos detectar tubos o patas)
                aspect_ratio = w / float(h)
                if 0.5 < aspect_ratio < 2.0:
                    if area > max_area:
                        max_area = area
                        best_box = (x, y, w, h)

        return best_box, mask

    def run(self):
        cv2.namedWindow("Vision Brain - Analisis RGB", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Vision Brain - Mascara Negra", cv2.WINDOW_AUTOSIZE)

        print("[INFO] Bucle de razonamiento iniciado. Pulsa 'q' para salir.")
        
        while self.running:
            if self.frame_rgb is None:
                time.sleep(0.1)
                continue

            # Hacemos una copia para dibujar la interfaz sin alterar el original
            display_frame = self.frame_rgb.copy()
            
            # 1. Buscar la caja en la imagen RGB
            bbox, mask = self.detect_black_box(self.frame_rgb)

            if bbox is not None:
                x, y, w, h = bbox
                
                # Dibujar bounding box
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # 2. Calcular la Profundidad real de la caja
                box_z = self.get_robust_depth(x, y, w, h)
                
                if box_z is not None:
                    # 3. Calcular la desviación lateral (X del robot, Y de la imagen)
                    x_c = int(x + w / 2)
                    y_c = int(y + h / 2)
                    cv2.circle(display_frame, (x_c, y_c), 5, (0, 0, 255), -1)
                    
                    # Matemática de trigonometría básica de cámara
                    desviacion_lateral = ((x_c - CX) * box_z) / FOCAL_LENGTH
                    
                    # HUD Visual
                    cv2.putText(display_frame, f"Z_Frontal: {box_z:.2f} m", (x, y - 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Desviacion Y: {desviacion_lateral:.2f} m", (x, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                    
                    # 4. Lógica de Razonamiento (Qué haría el robot si pudiera moverse)
                    if box_z > 0.8:
                        estado = "ACERCARSE (w)"
                    elif abs(desviacion_lateral) > 0.10:
                        estado = "ORBITAR/CENTRAR (a/d)"
                    else:
                        estado = "DISTANCIA DE AGARRE OPTIMA"
                        
                    cv2.putText(display_frame, f"DECISION: {estado}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Z_Frontal: CIEGO (Absorcion IR)", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "ESTADO: BUSCANDO CAJA...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Mostrar ventanas
            cv2.imshow("Vision Brain - Analisis RGB", display_frame)
            cv2.imshow("Vision Brain - Mascara Negra", mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    brain = VisionBrainTester()
    brain.run()
