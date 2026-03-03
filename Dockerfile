FROM ros:humble-ros-base

ENV DEBIAN_FRONTEND=noninteractive

# Añadimos TODAS las dependencias que hemos usado hoy
RUN apt-get update && apt-get install -y \
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
    && rm -rf /var/lib/apt/lists/*

RUN rosdep update

WORKDIR /root/ros2_ws

# Esto carga automáticamente el entorno al entrar
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "if [ -f /root/ros2_ws/install/setup.bash ]; then source /root/ros2_ws/install/setup.bash; fi" >> ~/.bashrc

CMD ["bash"]
