import tkinter as tk
from tkinter import ttk
import socket, threading, zmq, cv2, numpy as np
from PIL import Image, ImageTk
import time
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage


# Configuración para el simulador local
ROBOT_IP = '127.0.0.1'
CAM_PORT = "5555"      # Puerto de la cámara de MuJoCo
CMD_PORT = 6000        # Puerto de comandos del puente/simulador

class G1SimRemoteControl:
    def __init__(self, root):
        
        self.root = root
        self.root.title("G1 Simulator Remote Control")
        self.root.geometry("800x700")
        self.root.configure(bg="#1a1a1a")

        # --- VIDEO ---
        # Título de la cámara
        tk.Label(self.root, text="LIVE SIMULATOR FEED (RealSense)", bg="#1a1a1a", fg="white", 
                 font=("Arial", 12, "bold")).pack(pady=5)
        
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(pady=5, fill=tk.BOTH, expand=True)

        # --- PANEL DE CONTROL (Locomoción) ---
        move_frame = tk.LabelFrame(self.root, text=" Locomotion (Keyboard) ", bg="#2c3e50", fg="white", font=("Arial", 10, "bold"))
        move_frame.pack(fill=tk.X, padx=20, pady=20)

        controls_text = (
            "W / S : Forward / Backward\n"
            "A / D : Strafe Left / Right\n"
            "Q / E : Rotate Left / Right"
        )
        tk.Label(move_frame, text=controls_text, bg="#2c3e50", fg="#ecf0f1", 
                 font=("Arial", 11), justify=tk.LEFT).pack(pady=15, padx=20)

        # --- STATUS BAR ---
        self.status = tk.StringVar(value="Connecting to Simulator...")
        tk.Label(self.root, textvariable=self.status, bg="#333", fg="#00FF00", font=("Consolas", 9)).pack(side=tk.BOTTOM, fill=tk.X)

        # --- KEYBOARD SETUP ---
        self.current_key = None
        self.release_timer = None
        self.valid_keys = ['w', 'a', 's', 'd', 'q', 'e']
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.bind('<KeyRelease>', self.on_key_release)
        self.root.focus_set()

        # --- RED ZMQ (Cámara Simulator) ---
        self.ctx = zmq.Context()
        self.sock_cam = self.ctx.socket(zmq.SUB)
        self.sock_cam.setsockopt(zmq.CONFLATE, 1)  # Mantener solo el último frame
        self.sock_cam.setsockopt_string(zmq.SUBSCRIBE, "")
        
        try: 
            self.sock_cam.connect(f"tcp://{ROBOT_IP}:{CAM_PORT}")
            self.status.set("Connected to Simulator Cam: Port 5555")
        except Exception as e: 
            self.status.set(f"ZMQ Connection Error: {e}")

        self.alive = True
        
        # Hilos secundarios
        threading.Thread(target=self.ping_loop, daemon=True).start()
        self.latest_frame = None # Este atributo almacenará el último frame recibido para usarlo en el NODO de ROS2
        self.update_img()

    def ping_loop(self):
        """Mantiene la conexión activa con el puente de simulación."""
        while self.alive:
            self.send('ping')
            time.sleep(0.5)

    def on_key_press(self, event):
        key = event.keysym.lower()
        if key not in self.valid_keys: return
        
        if self.release_timer is not None:
            self.root.after_cancel(self.release_timer)
            self.release_timer = None
            
        if self.current_key != key:
            self.current_key = key
            self.send(key)

    def on_key_release(self, event):
        key = event.keysym.lower()
        if key == self.current_key:
            # Pequeño delay para suavizar el stop en caso de pulsaciones rápidas
            self.release_timer = self.root.after(50, self.execute_stop)

    def execute_stop(self):
        self.current_key = None
        self.send('stop')
        self.release_timer = None

    def update_img(self):
        """Recibe el frame del simulador y lo muestra en el GUI."""
        if not self.alive: return

        try:
            # Recibir frame por ZMQ sin bloquear el GUI
            raw_audio = self.sock_cam.recv(zmq.NOBLOCK)
            frame_np = np.frombuffer(raw_audio, dtype=np.uint8)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

            if frame is not None:
                self.latest_frame = frame  # Guardar el último frame recibido para el nodo ROS2
                # Convertir de BGR a RGB para Tkinter
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                # Ajustar tamaño manteniendo proporción (puedes cambiar 640x480)
                img = ImageTk.PhotoImage(img.resize((640, 480), Image.Resampling.LANCZOS))
                
                self.video_label.configure(image=img)
                self.video_label.image = img
        except zmq.Again:
            pass # No hay frame nuevo aún
        except Exception as e:
            print(f"Error decodificando imagen: {e}")
        
        # Llamar de nuevo después de ~30ms (aprox 30 FPS)
        self.root.after(30, self.update_img)
    
    def get_latest_frame(self):
        """Devuelve el último frame recibido del simulador."""
        return self.latest_frame
        


    def send(self, cmd):
        """Envía comandos de texto al puente del simulador vía TCP."""
        if cmd != 'ping': 
            self.status.set(f"SIM CMD: {cmd.upper()}")
            
        # Ejecutar en hilo para no congelar la UI
        threading.Thread(target=self._worker_send, args=(cmd,), daemon=True).start()

    def _worker_send(self, cmd):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect((ROBOT_IP, CMD_PORT))
                s.sendall(cmd.encode('utf-8'))
                # No esperamos respuesta larga en simulador para ganar velocidad
                s.recv(128) 
        except:
            pass


class G1SimNode(Node):
    """Nodo de ROS2 que recibe comando de cmd_vel, y usa la función de enviar del GUI para transmitir al simulador."""
    def __init__(self,send_function,latest_frame_function):
        super().__init__('g1_sim_node')
        self.get_logger().info("G1 Simulator Node Initialized")
        self.subscription = self.create_subscription(Twist, "cmd_vel", self.cmd_callback, 10)
        self.subscription  # Evitar advertencia de variable no utilizada
        self.send_function = send_function
        self.latest_frame_function = latest_frame_function
        self.publisher = self.create_publisher(CompressedImage, '/simulation/image_compressed', 10)  
        self.timer = self.create_timer(0.1, self.publish_image)  # Publicar cada 100ms, si no creo el timer no publica nada
        
    
    def publish_image(self):
        """Publica un frame de imagen comprimida en el tópico ROS2."""
        frame = self.latest_frame_function()  # Obtener el último frame pasado por función del GUI
        if frame is None:
            self.get_logger().warn("No frame received")
            return  # No hay frame disponible
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()

        self.publisher.publish(msg)

    "función de cmd_vel, cambiar para la sensibilidad"
    def cmd_callback(self, msg):
        linear_x = msg.linear.x
        linear_y = msg.linear.y
        angular_z = msg.angular.z
        
        if linear_x > 0:
            command = 'w'
        elif linear_x < 0:
            command = 's'
        elif angular_z > 0:
            command = 'q'
        elif angular_z < 0:
            command = 'e'
        elif linear_y > 0:
            command = 'a'
        elif linear_y < 0:
            command = 'd'
        else:
            command = 'stop'
        
        # Enviar el comando al simulador
        self.send_function(command)



if __name__ == "__main__":
    rclpy.init()

    root = tk.Tk()

    app = G1SimRemoteControl(root)

    sim_node = G1SimNode(app.send,app.get_latest_frame) #este es el nodo de ROS2, que hereda la funcion de enviar datos

    threading.Thread(
        target=rclpy.spin,
        args=(sim_node,),
        daemon=True
    ).start()
    def on_closing():
        app.alive = False
        root.destroy()
        os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()
