# G1UR Manipulation & Navigation 🤖

Este repositorio contiene el espacio de trabajo de ROS 2 (Humble) para el control, navegación y visión del Unitree G1. Está diseñado para ejecutarse dentro de un contenedor Docker directamente en el hardware del robot, manteniendo el sistema host limpio y asegurando la compatibilidad de dependencias (como CycloneDDS, OpenCV y el Unitree SDK2).

## 🚀 Requisitos y Configuración Inicial (En el Robot)

Para aislar el entorno, usamos una imagen de Docker personalizada (`humble_g1`) que ya contiene todas las librerías necesarias compiladas para ARM64.

### 1. Alias de Docker
Para facilitar el arranque del entorno, añade este alias al archivo `~/.bashrc` del robot:

`alias docker_humble='sudo docker run -it --rm --name entorno_humble --net=host --privileged -v /dev:/dev -v ~/robot_ws:/root/ros2_ws humble_g1'`

*Asegúrate de ejecutar `source ~/.bashrc` después de añadirlo.*

### 2. Sincronización del Repositorio
El flujo de trabajo consiste en programar en local (PC) y ejecutar en el robot. Para actualizar el código en el robot:
`cd ~/robot_ws`
`git pull`

*(Nota: Si hay problemas de permisos creados por Docker, ejecuta `sudo chown -R $USER:$USER ~/robot_ws` antes de hacer el pull).*

---

## 🛠️ Flujo de Trabajo Diario

**1. Arrancar el Entorno (Robot)**
`docker_humble`

Esto abrirá una terminal interactiva como `root` dentro de `/root/ros2_ws`, con el volumen mapeado a los archivos de este repositorio y la red compartida con el host (`--net=host`).

**2. Compilar el Workspace (Dentro de Docker)**
Si has modificado archivos C++ en la carpeta `src/`:
`colcon build --symlink-install`
`source install/setup.bash`

**3. Lanzar Nodos**
Por ejemplo, para ejecutar el script que extrae el vídeo de la RealSense usando el SDK interno sin bloquear la App de Unitree:
`python3 /root/ros2_ws/sdk_camera_ros.py`

## 📡 Visualización en el PC Local

No es necesario usar `ssh -X` ni VNC. Como el Docker usa `--net=host`, los topics de ROS 2 son visibles en la red WiFi.
1. Asegúrate de estar en la misma red que el robot.
2. Abre `rviz2` en tu ordenador local.
3. Para ver la cámara sin lag, añade un display de tipo **Image**, selecciona el topic `/camera/image/compressed` y asegúrate de configurar el **QoS Profile (Reliability Policy)** a **Best Effort**.
