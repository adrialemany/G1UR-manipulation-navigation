FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Añadimos TODAS las dependencias que hemos usado hoy y las extraídas del entorno
RUN apt-get update && apt-get install -y \
    # Dependencias originales de tu Dockerfile
    git build-essential cmake \
    python3-colcon-common-extensions python3-rosdep python3-vcstool wget \
    ros-humble-rviz2 \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions \
    ros-humble-rosidl-default-generators \
    ros-humble-rosidl-generator-dds-idl \
    python3-opencv python3-numpy \
    # --- NUEVAS DEPENDENCIAS AÑADIDAS ---
    # Herramientas de sistema, red y Python extra
    python3-pip \
    iproute2 \
    net-tools \
    nano \
    curl \
    # Herramientas de Audio y Vídeo (Profesor Enric)
    alsa-utils \
    ffmpeg \
    v4l-utils \
    # Paquetes ROS 2 de Visión y Cámaras
    ros-humble-v4l2-camera \
    ros-humble-realsense2-camera \
    ros-humble-realsense2-description \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport \
    ros-humble-vision-msgs \
    # Almacenamiento y Rosbags
    ros-humble-rosbag2-storage-mcap \
    # Middleware y Red DDS
    ros-humble-cyclonedds \
    ros-humble-rmw-cyclonedds-cpp \
    # Navegación y Planificación (Nav2 y MoveIt)
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-moveit \
    # --- DEPENDENCIAS PARA SLAM 3D (FAST-LIO) ---
    libpcl-dev \
    && rm -rf /var/lib/apt/lists/*

# Descargar, compilar e instalar Livox SDK2 (Requisito para el LiDAR)
WORKDIR /root
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
    cd Livox-SDK2 && \
    mkdir build && cd build && \
    cmake .. && make -j && \
    make install && \
    # Limpiamos el código fuente para no engordar la imagen Docker a lo tonto
    rm -rf /root/Livox-SDK2

RUN rosdep update

WORKDIR /root/ros2_ws

# Esto carga automáticamente el entorno al entrar
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "if [ -f /root/ros2_ws/install/setup.bash ]; then source /root/ros2_ws/install/setup.bash; fi" >> ~/.bashrc

CMD ["bash"]
