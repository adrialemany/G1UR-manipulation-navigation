# SLAM Toolbox — Integración con G1 MuJoCo

## Archivos añadidos

```
g1man/
├── mapeo/
│   └── slam_toolbox/
│       ├── slam_toolbox_params.yaml   ← Configuración completa con loop closure
│       └── g1_slam.launch.py          ← Launch file (online_async)
└── rviz2/
    └── g1_slam.rviz                   ← Visualización con frame 'map'
```

## Instalación (una sola vez)

```bash
sudo apt install ros-humble-slam-toolbox
```

## Cómo usarlo

### Modo mapping (construir mapa + loop closure)

Abre 4 terminales con ROS 2 Humble activado:

```bash
# Terminal 1 — Simulación MuJoCo
cd mujoco/simulacion && python3 run_sim_ai_g1.py

# Terminal 2 — Bridge LiDAR → ROS 2
cd mujoco/simulacion && python3 mujoco_ros2_lidar_bridge.py

# Terminal 3 — SLAM Toolbox
ros2 launch mapeo/slam_toolbox/g1_slam.launch.py

# Terminal 4 — Visualización
rviz2 -d rviz2/g1_slam.rviz

# Terminal 5 — Teleop para mover el robot
python3 teleop/g1_client_mujoco.py
```

### Guardar el mapa

Una vez explorado el entorno:

```bash
# Formato PGM+YAML → compatible con Nav2 map_server / AMCL
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
  "{name: {data: 'maps/mi_mapa'}}"
# Genera: maps/mi_mapa.pgm + maps/mi_mapa.yaml

# Formato nativo → permite reanudar el SLAM desde donde lo dejaste
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
  "{filename: {data: 'maps/mi_mapa'}}"
# Genera: maps/mi_mapa.posegraph + maps/mi_mapa.data
```

### Modo localización (mapa ya hecho)

```bash
ros2 launch mapeo/slam_toolbox/g1_slam.launch.py \
  mode:=localization \
  map:=maps/mi_mapa     # sin extensión, apunta al .posegraph
```

---

## Por qué funciona sin tocar el bridge

SLAM Toolbox necesita:
- `/scan` → ya lo publica `mujoco_ros2_lidar_bridge.py` (frame: `lidar_link`)
- `/odom` → ya lo publica el bridge (frame: `odom → base_link`)
- TF: `odom → base_link → lidar_link` → ya lo publica el bridge

SLAM Toolbox **añade** el transform `map → odom` al árbol TF.
El bridge ya tiene el comentario correcto: no publica `world → odom`,
así que no hay conflicto de TF.

### Árbol TF resultante

```
map          ← SLAM Toolbox lo publica
 └── odom    ← bridge (estático identidad implícito)
      └── base_link   ← bridge (pose del pelvis desde MuJoCo)
           └── lidar_link  ← bridge (pose relativa del LiDAR)
```

---

## Diferencia con el mapper anterior (`mujoco_slam_mapper.py`)

| Característica           | `mujoco_slam_mapper.py`   | SLAM Toolbox             |
|--------------------------|---------------------------|--------------------------|
| Loop closure             | ❌ No                      | ✅ Sí (Ceres solver)      |
| Corrección de deriva     | ❌ No                      | ✅ Sí (pose graph)        |
| Guardar/reanudar sesión  | Solo PGM/YAML             | PGM/YAML + posegraph     |
| Modo solo localización   | ❌ No                      | ✅ Sí                     |
| Compatible con Nav2      | ✅ Sí (genera PGM/YAML)    | ✅ Sí (nativo)            |
| Frame publicado          | `/map` en frame `odom`    | `/map` en frame `map`    |
| Complejidad              | Bajo (Python puro)        | Media (nodo C++)         |

---

## Usar el mapa de SLAM Toolbox con Nav2

El mapa guardado como PGM+YAML es directamente compatible con el
`g1_nav2.launch.py` existente:

```bash
ros2 launch navegacion/nav2/g1_nav2.launch.py \
  map:=$(pwd)/maps/mi_mapa.yaml
```

La diferencia es que ahora el mapa tendrá las correcciones de loop closure
aplicadas, por lo que AMCL tendrá una referencia mucho más precisa.
