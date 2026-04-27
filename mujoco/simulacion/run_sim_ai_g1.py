#!/usr/bin/env python3
"""
run_sim_ai_g1.py  —  Motor de simulación + control unificado del G1
====================================================================

Sistema de control por PRIORIDAD:
  Prioridad 1 (más alta): Nav2  → /cmd_vel  (ROS 2 Twist)
  Prioridad 2 (más baja):  Teleop → TCP:6000  (protocolo multi-tecla)

Cuando Nav2 está activo (recibe /cmd_vel con frecuencia), la teleop
queda BLOQUEADA. Al dejar de llegar mensajes de Nav2 (> NAV2_TIMEOUT s),
el control vuelve automáticamente a teleop.

Protocolo teleop TCP (puerto 6000)
-----------------------------------
El cliente envía un JSON con el conjunto de teclas presionadas en ese
instante, e.g.:  {"keys": ["w", "d"]}
Se admiten combinaciones de cualquier número de teclas simultáneas.
Teclas:  w/s = vx+/-   a/d = vy+/-   q/e = yaw+/-   stop = parar todo

Esto es retrocompatible con el cliente anterior siempre que el cliente
mande strings simples (se detectan y se envuelven en el formato nuevo
automáticamente).
"""

# ── Entorno ─────────────────────────────────────────────────────────────────
import os
os.environ["CYCLONEDDS_URI"] = """<CycloneDDS>
    <Domain>
        <SharedMemory>
            <Enable>false</Enable>
        </SharedMemory>
    </Domain>
</CycloneDDS>"""

import time
import math
import json
import socket
import threading
import subprocess
import numpy as np
import onnxruntime as ort

from unitree_sdk2py.core.channel import (ChannelFactoryInitialize,
                                          ChannelPublisher, ChannelSubscriber)
from unitree_sdk2py.idl.default import (unitree_hg_msg_dds__LowState_,
                                         unitree_hg_msg_dds__LowCmd_)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_


# ════════════════════════════════════════════════════════════════════════════
# Configuración
# ════════════════════════════════════════════════════════════════════════════

TCP_PORT        = 6000          # puerto teleop
ARM_UDP_PORT    = 9876          # puerto brazos
RESET_UDP_PORT  = 6005          # puerto reset MuJoCo

# Velocidades máximas (deben coincidir con nav2_params.yaml)
VX_MAX   =  0.6
VX_MIN   = -0.4
VY_MAX   =  0.3
VY_MIN   = -0.3
YAW_MAX  =  0.8

# Velocidades teleop (teclas)
VX_FWD   =  0.6
VX_BCK   = -0.4
VY_LEFT  =  0.3
VY_RIGHT = -0.3
YAW_L    =  0.6
YAW_R    = -0.6

# Nav2: si no llega ningún /cmd_vel en este tiempo → ceder a teleop
NAV2_TIMEOUT = 0.5   # segundos

# Control loop
CONTROL_HZ = 50


# ════════════════════════════════════════════════════════════════════════════
# Estado global compartido — acceso protegido por lock
# ════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()

# Comandos finales que lee el bucle de control
comandos = {'vx': 0.0, 'vy': 0.0, 'yaw': 0.0}

# Comando deseado por cada fuente
_nav2_cmd    = {'vx': 0.0, 'vy': 0.0, 'yaw': 0.0}
_teleop_cmd  = {'vx': 0.0, 'vy': 0.0, 'yaw': 0.0}

_nav2_last_t   = 0.0   # timestamp del último /cmd_vel recibido
_nav2_active   = False  # True si Nav2 tiene el control ahora mismo

low_state          = None
external_arm_targets = {}


def _nav2_is_active() -> bool:
    """Devuelve True si Nav2 está enviando comandos activamente."""
    return (time.time() - _nav2_last_t) < NAV2_TIMEOUT


def _merge_commands():
    """
    Aplica la política de prioridad y escribe en `comandos`.
    Llamar siempre dentro de _lock.
    """
    global _nav2_active
    if _nav2_is_active():
        _nav2_active = True
        comandos['vx']  = _nav2_cmd['vx']
        comandos['vy']  = _nav2_cmd['vy']
        comandos['yaw'] = _nav2_cmd['yaw']
    else:
        if _nav2_active:
            # Nav2 acaba de perder el control → parar primero
            _teleop_cmd['vx'] = _teleop_cmd['vy'] = _teleop_cmd['yaw'] = 0.0
        _nav2_active = False
        comandos['vx']  = _teleop_cmd['vx']
        comandos['vy']  = _teleop_cmd['vy']
        comandos['yaw'] = _teleop_cmd['yaw']


# ════════════════════════════════════════════════════════════════════════════
# Política (RL)
# ════════════════════════════════════════════════════════════════════════════

class HolosomaLocomotion:
    def __init__(self, model_path):
        self.session    = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        self.default_angles = np.zeros(29, dtype=np.float32)
        self.default_angles[[0, 6]]   = -0.312
        self.default_angles[[3, 9]]   =  0.669
        self.default_angles[[4, 10]]  = -0.363
        self.default_angles[[15, 22]] =  0.2
        self.default_angles[16]       =  0.2
        self.default_angles[23]       = -0.2
        self.default_angles[[18, 25]] =  0.6

        self.last_action  = np.zeros(29, dtype=np.float32)
        self.control_freq = CONTROL_HZ
        self.gait_period  = 1.0
        self.phase_dt     = 2 * math.pi / (self.control_freq * self.gait_period)
        self.phase        = np.array([0.0, math.pi], dtype=np.float32)

    def get_target_positions(self, state, cmd, ext_targets):
        cmd_mag = math.sqrt(cmd['vx']**2 + cmd['vy']**2 + cmd['yaw']**2)
        if cmd_mag < 0.01:
            self.phase = np.array([math.pi, math.pi], dtype=np.float32)
        else:
            self.phase = (self.phase + self.phase_dt) % (2 * math.pi)

        jp = np.array(state['joint_pos'])
        jv = np.array(state['joint_vel'])
        for i in ext_targets:
            jp[i] = self.default_angles[i] + (self.last_action[i] * 0.25)
            jv[i] = 0.0

        obs = np.zeros(100, dtype=np.float32)
        obs[0:29]  = self.last_action
        obs[29:32] = np.array(state['gyro']) * 0.25
        obs[32]    = cmd['yaw']
        obs[33:35] = [cmd['vx'], cmd['vy']]
        obs[35:37] = np.cos(self.phase)
        obs[37:66] = jp - self.default_angles
        obs[66:95] = jv * 0.05
        obs[95:98] = np.array(state['gravity'])
        obs[98:100]= np.sin(self.phase)

        action = self.session.run(
            None, {self.input_name: np.expand_dims(obs, 0)})[0].squeeze()
        self.last_action = action
        return self.default_angles + (action * 0.25)


# ════════════════════════════════════════════════════════════════════════════
# Callbacks DDS
# ════════════════════════════════════════════════════════════════════════════

def state_callback(msg: LowState_):
    global low_state
    low_state = msg


def quaternion_to_gravity(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return [
        -2 * (x*z - w*y),
        -2 * (y*z + w*x),
        -(1 - 2*(x*x + y*y))
    ]


# ════════════════════════════════════════════════════════════════════════════
# Listener Nav2  —  /cmd_vel  (ROS 2)
# ════════════════════════════════════════════════════════════════════════════

def nav2_listener():
    """
    Suscriptor ROS 2 a /cmd_vel.
    Escribe directo en comandos (igual que el original) + actualiza
    _nav2_last_t para el sistema de prioridad.
    """
    global _nav2_last_t

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from geometry_msgs.msg import Twist
    except ImportError:
        print("[WARN] rclpy no disponible — /cmd_vel deshabilitado")
        return

    if not rclpy.ok():
        rclpy.init(args=None)

    node = rclpy.create_node('g1_nav2_priority_bridge')

    # QoS RELIABLE para coincidir con Nav2 velocity_smoother
    qos = QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
    )

    def cb(msg: Twist):
        global _nav2_last_t
        vx  = float(np.clip(msg.linear.x,  VX_MIN,  VX_MAX))
        vy  = float(np.clip(msg.linear.y,  VY_MIN,  VY_MAX))
        yaw = float(np.clip(msg.angular.z, -YAW_MAX, YAW_MAX))
        _nav2_last_t = time.time()
        with _lock:
            # Nav2 escribe directo — tiene prioridad absoluta
            comandos['vx']  = vx
            comandos['vy']  = vy
            comandos['yaw'] = yaw
        print(f"[NAV2] vx={vx:.2f}  vy={vy:.2f}  yaw={yaw:.2f}")

    node.create_subscription(Twist, '/cmd_vel', cb, qos)
    print("[INFO] Nav2 bridge activo — suscrito a /cmd_vel (RELIABLE)")
    rclpy.spin(node)


# ════════════════════════════════════════════════════════════════════════════
# Listener Teleop  —  TCP:6000  (multi-tecla JSON)
# ════════════════════════════════════════════════════════════════════════════
#
# Protocolo aceptado (retrocompatible):
#
#   Nuevo (multi-tecla):  {"keys": ["w", "d"]}
#   Antiguo (una tecla):  "w"  /  "stop"  /  "ping"
#
# El servidor es persistente: el cliente puede mantener la conexión abierta
# y mandar frames JSON separados por '\n', O bien una conexión por mensaje
# (modo antiguo).  Ambos funcionan.
# ════════════════════════════════════════════════════════════════════════════

def _keys_to_cmd(keys: list) -> dict:
    """Convierte un conjunto de teclas en velocidades vx/vy/yaw."""
    vx = vy = yaw = 0.0

    if 'w' in keys: vx  += VX_FWD
    if 's' in keys: vx  += VX_BCK
    if 'a' in keys: vy  += VY_LEFT
    if 'd' in keys: vy  += VY_RIGHT
    if 'q' in keys: yaw += YAW_L
    if 'e' in keys: yaw += YAW_R

    # Clamp por si se suman w+s o a+d a la vez
    vx  = float(np.clip(vx,  VX_MIN,  VX_MAX))
    vy  = float(np.clip(vy,  VY_MIN,  VY_MAX))
    yaw = float(np.clip(yaw, -YAW_MAX, YAW_MAX))

    return {'vx': vx, 'vy': vy, 'yaw': yaw}


def _handle_client(conn):
    """Maneja una conexión TCP del cliente de teleop."""
    try:
        conn.settimeout(2.0)
        buf = b""
        while True:
            try:
                chunk = conn.recv(1024)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk

            # Intentar procesar mensajes completos (separados por \n o EOF)
            while True:
                # Buscar separador de mensaje
                nl = buf.find(b'\n')
                if nl >= 0:
                    raw = buf[:nl].strip()
                    buf = buf[nl+1:]
                else:
                    # Sin \n: tratar todo el buffer como un mensaje si parece completo
                    raw = buf.strip()
                    buf = b""

                if not raw:
                    break

                try:
                    text = raw.decode('utf-8')
                except UnicodeDecodeError:
                    break

                # ── Parsear mensaje ──────────────────────────────────────────
                if text in ('stop', 'ping'):
                    with _lock:
                        _teleop_cmd['vx'] = _teleop_cmd['vy'] = _teleop_cmd['yaw'] = 0.0
                        _merge_commands()
                    conn.sendall(b"OK\n")
                    if nl < 0:
                        break   # mensaje sin \n → asumir conexión de un solo uso
                    continue

                # Intentar JSON nuevo: {"keys": [...]}
                parsed_json = False
                if text.startswith('{'):
                    try:
                        data = json.loads(text)
                        keys = [str(k).lower() for k in data.get('keys', [])]
                        cmd  = _keys_to_cmd(keys)
                        nav2_bloqueado = _nav2_is_active()
                        if not nav2_bloqueado:
                            with _lock:
                                _teleop_cmd.update(cmd)
                                _merge_commands()
                        status = b"OK\n" if not nav2_bloqueado else b"BLOCKED_NAV2\n"
                        conn.sendall(status)
                        parsed_json = True
                    except (json.JSONDecodeError, Exception):
                        pass

                if not parsed_json:
                    # Compatibilidad con cliente antiguo (tecla suelta)
                    key = text.lower().strip()
                    if key in ('w', 's', 'a', 'd', 'q', 'e'):
                        cmd = _keys_to_cmd([key])
                        nav2_bloqueado = _nav2_is_active()
                        if not nav2_bloqueado:
                            with _lock:
                                _teleop_cmd.update(cmd)
                                _merge_commands()
                        conn.sendall(b"OK\n")
                    else:
                        conn.sendall(b"UNKNOWN\n")

                if nl < 0:
                    break  # conexión de un solo uso (modo antiguo)

    except Exception:
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def teleop_listener():
    """Servidor TCP multi-tecla para teleop. Retrocompatible con cliente antiguo."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('0.0.0.0', TCP_PORT))
    srv.listen(10)
    print(f"[INFO] Teleop TCP escuchando en :{TCP_PORT} (multi-tecla JSON + modo clásico)")
    while True:
        try:
            conn, _ = srv.accept()
            threading.Thread(target=_handle_client, args=(conn,), daemon=True).start()
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════════════════
# Watchdog de Nav2  —  imprime estado en consola
# ════════════════════════════════════════════════════════════════════════════

def nav2_watchdog():
    prev_active = False
    while True:
        time.sleep(0.2)
        active = _nav2_is_active()
        if active != prev_active:
            if active:
                print("\n[🟢 NAV2] Control ACTIVO — teleop bloqueada")
            else:
                print("\n[🔴 NAV2] Control INACTIVO — teleop libre")
            prev_active = active


# ════════════════════════════════════════════════════════════════════════════
# Listener brazos  —  UDP:9876
# ════════════════════════════════════════════════════════════════════════════

def arm_listener():
    global external_arm_targets
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', ARM_UDP_PORT))
    sock.settimeout(0.5)
    while True:
        try:
            data, _ = sock.recvfrom(1024)
            incoming = json.loads(data.decode('utf-8'))
            external_arm_targets = {int(k): float(v) for k, v in incoming.items()}
        except Exception:
            continue


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "fastsac_g1_29dof.onnx")

    print("[INFO] Lanzando unitree_mujoco.py...")
    sim_proc = subprocess.Popen(["python3", "unitree_mujoco.py"], cwd=script_dir)
    time.sleep(1.0)

    # Arrancar listeners en hilos
    threading.Thread(target=teleop_listener,  daemon=True).start()
    threading.Thread(target=arm_listener,     daemon=True).start()
    threading.Thread(target=nav2_listener,    daemon=True).start()
    threading.Thread(target=nav2_watchdog,    daemon=True).start()

    # DDS
    ChannelFactoryInitialize(1, "lo")
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(state_callback, 10)
    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    controller = HolosomaLocomotion(model_path=model_path)

    kp = ([40.18, 99.10, 40.18, 99.10, 28.50, 28.50] * 2
          + [40.18, 28.50, 28.50]
          + [14.25, 14.25, 14.25, 14.25, 16.78, 16.78, 16.78] * 2)
    kd = ([2.56, 6.31, 2.56, 6.31, 1.81, 1.81] * 2
          + [2.56, 1.81, 1.81]
          + [0.91, 0.91, 0.91, 0.91, 1.07, 1.07, 1.07] * 2)

    print("[INFO] Esperando datos del simulador...")
    while low_state is None:
        time.sleep(0.1)

    print("[INFO] ¡Control activo!")
    print("       • Teleop:  cliente TCP en :6000  (WASD/QE, multi-tecla JSON)")
    print("       • Nav2:    suscrito a /cmd_vel   (prioridad absoluta)")

    try:
        while True:
            t0 = time.time()

            estado = {
                'gyro':      low_state.imu_state.gyroscope,
                'gravity':   quaternion_to_gravity(low_state.imu_state.quaternion),
                'joint_pos': [low_state.motor_state[i].q  for i in range(29)],
                'joint_vel': [low_state.motor_state[i].dq for i in range(29)],
            }

            # Detección de caída
            if estado['gravity'][2] > -0.5:
                print("🚨 ¡Caída detectada! Reseteando...")
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.sendto(b"reset", ("127.0.0.1", RESET_UDP_PORT))
                    s.close()
                except Exception:
                    pass
                with _lock:
                    comandos['vx'] = comandos['vy'] = comandos['yaw'] = 0.0
                    _nav2_cmd['vx']   = _nav2_cmd['vy']   = _nav2_cmd['yaw']   = 0.0
                    _teleop_cmd['vx'] = _teleop_cmd['vy'] = _teleop_cmd['yaw'] = 0.0
                controller.phase = np.array([0.0, math.pi], dtype=np.float32)
                time.sleep(0.1)
                continue

            # Leer comandos (snapshot thread-safe)
            with _lock:
                cmd_snap = dict(comandos)

            # ── Heading automático (solo Nav2) ──────────────────────────────
            # Cuando Nav2 controla el robot, convierte vy en yaw para que el
            # robot mire hacia donde se desplaza (comportamiento omnidireccional).
            # En teleop manual se desactiva para que A/D sean strafe puro.
            vx = cmd_snap['vx']
            vy = cmd_snap['vy']
            speed_xy = math.sqrt(vx**2 + vy**2)
            if _nav2_is_active() and speed_xy > 0.05 and abs(cmd_snap['yaw']) < 0.1:
                desired_heading = math.atan2(vy, vx)
                heading_gain = 1.2
                cmd_snap['yaw'] = float(np.clip(
                    desired_heading * heading_gain,
                    -YAW_MAX, YAW_MAX
                ))

            targets = controller.get_target_positions(estado, cmd_snap, external_arm_targets)

            cmd_msg = unitree_hg_msg_dds__LowCmd_()
            for i in range(29):
                cmd_msg.motor_cmd[i].mode = 1
                cmd_msg.motor_cmd[i].dq   = 0.0
                cmd_msg.motor_cmd[i].kp   = kp[i]
                cmd_msg.motor_cmd[i].kd   = kd[i]
                cmd_msg.motor_cmd[i].tau  = 0.0
                cmd_msg.motor_cmd[i].q    = (external_arm_targets[i]
                                              if i in external_arm_targets
                                              else targets[i])
            pub.Write(cmd_msg)

            elapsed = time.time() - t0
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - elapsed))

    except KeyboardInterrupt:
        print("\n[INFO] Detenido.")
        sim_proc.terminate()
