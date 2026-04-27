#!/usr/bin/env python3
"""
g1_slam.launch.py
=================
Lanza SLAM Toolbox en modo online_async para mapear en tiempo real
con cierre de bucle automático.

Flujo completo (ejecutar desde la raíz del proyecto):
  Terminal 1 → python3 mujoco/simulacion/run_sim_ai_g1.py
  Terminal 2 → python3 mujoco/simulacion/mujoco_ros2_lidar_bridge.py
  Terminal 3 → ros2 launch mapeo/slam_toolbox/g1_slam.launch.py
  Terminal 4 → rviz2 -d rviz2/g1_slam.rviz
  Terminal 5 → python3 teleop/g1_client_mujoco.py

Guardar el mapa una vez explorado:
  # Formato PGM+YAML (compatible con Nav2 map_server):
  ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \\
    "{name: {data: 'maps/mi_mapa'}}"

  # Formato nativo (para reanudar el SLAM más tarde):
  ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \\
    "{filename: {data: 'maps/mi_mapa'}}"

Uso con mapa previo (modo localization):
  ros2 launch mapeo/slam_toolbox/g1_slam.launch.py \\
    mode:=localization map:=maps/mi_mapa
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_params = os.path.join(this_dir, 'slam_toolbox_params.yaml')

    # ----- Argumentos -----
    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Ruta al slam_toolbox_params.yaml')

    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='mapping',
        description='"mapping" para construir mapa, "localization" para solo localizar')

    map_arg = DeclareLaunchArgument(
        'map',
        default_value='',
        description='Ruta al mapa previo .posegraph (solo en modo localization)')

    params_file = LaunchConfiguration('params_file')

    # ----- Nodo SLAM Toolbox (online async) -----
    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            params_file,
            # Sobreescribir modo si se pasó como argumento de lanzamiento
            {'mode': LaunchConfiguration('mode')},
            {'map_file_name': LaunchConfiguration('map')},
        ],
        # No remappings necesarios: el bridge ya publica /scan con el
        # frame correcto (lidar_link) y el TF tree está bien configurado
    )

    return LaunchDescription([
        params_arg,
        mode_arg,
        map_arg,
        slam_node,
    ])
