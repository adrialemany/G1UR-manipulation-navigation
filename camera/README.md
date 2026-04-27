# Unitree G1 Remote Camera Client

This directory contains a client-server architecture (`g1_client.py` and `g1_server.py`) for connecting to and monitoring the camera and audio streams of a Unitree G1 over a local network.

![g1 client](../assets/client.png)

## System Overview
1. **Server (`g1_server.py`):** Runs on the Unitree G1 robot. Captures frames from the RealSense camera and a secondary USB webcam, compresses them, and broadcasts them via ZeroMQ (ZMQ).
2. **Client (`g1_client.py`):** Runs on your local machine. Subscribes to the ZMQ streams and renders the video in a Tkinter GUI. Features a Text-To-Speech (TTS) panel for audio playback on the robot.

## Prerequisites

Install the required dependencies on your **local machine**:

```bash
sudo apt-get install libportaudio2
pip install pyzmq opencv-python Pillow numpy sounddevice
```

## Step-by-Step Usage

1. **Start the Server on the Robot**
Ensure the Unitree G1 is powered on and connected to your local network.
Note: If using an external USB camera, ensure it is plugged in before booting the robot.

SSH into the robot (replace <ROBOT_IP> with the actual IP, e.g., 192.168.1.126):

```bash
ssh unitree@<ROBOT_IP>
```

Once inside the robot, navigate to this directory and run the server:

```bash
python3 g1_server.py
```

2. **Start the Client Locally**
On your local workstation, open a new terminal, navigate to the camera directory, and run:

```bash
# Ensure you edit g1_client.py to match the robot's IP before running
python3 g1_client.py
```

Use the dropdown menu at the top of the interface to switch between the RealSense camera (Port 6001) and the USB Camera (Port 6002).

## Locomotion and Safety Note
The interface includes buttons for locomotion (Zero Torque, Damping, Squat, Stand). Always follow the correct state machine transitions. To use custom programmed arm movements, the robot must be placed in Rack mode. Ensure the area around the robot is clear before triggering any locomotion commands.

## On Simulation
To use the Mujoco Simulation instead of the physical robot, please read the [Mujoco instructions](../mujoco/README.md).

## Emotion Recognition & Robot Reaction (AI Integration)

This module integrates a multimodal Deep Learning model (Audio + Vision) to detect human emotions and trigger specific robot animations or inverse kinematics (IK) movements. This was developed in collaboration with the Korean research team.

### Files Overview
* **`model.py` & `inference.py`**: Contains the PyTorch architecture (`AVFusionModel`). It extracts audio features using Wav2Vec2 and visual features using a Swin Transformer + BiGRU, fusing them via Cross-Attention to predict among 6 emotion classes.
* **`controller.py`**: The main execution script for the **physical robot**. It captures face crops via `insightface` and audio via ZMQ, runs inference every few seconds, and triggers `G1ArmActionClient` prefabricated actions based on the detected emotion.
* **`emotions_g1_mujoco.py`**: The counterpart for the **MuJoCo simulation**. Instead of relying on internal robot macros, it uses Pinocchio-based Forward/Inverse Kinematics with Null-Space Projection to procedurally generate smooth arm animations corresponding to specific emotions (Happy, Sad, Angry, etc.).

### Prerequisites for Emotion Inference
To run the AI models and the IK solver, you need to install additional heavy dependencies on your local machine:

```bash
# Install Deep Learning and Vision dependencies
pip install torch torchvision torchaudio timm transformers insightface onnxruntime

# Install Pinocchio for Inverse Kinematics (Simulation script)
sudo apt install ros-humble-pinocchio
```

### Execution

1. **Start the Emotion Engine (IK Solver)**
Depending on your environment, run the corresponding IK engine script. This script opens a background UDP port (5005) waiting for emotion commands.

For simulation:

```bash
python3 emotions_g1_mujoco.py
```

For the physical robot:

```bash
python3 emotions_g1.py
```

*Note*: You can manually test the IK engine by typing an emotion directly into its terminal once it syncs.

2. **Run the AI Inference Controller**
In a new terminal, start the inference script. The system will look for faces, listen to the audio buffer, predict the emotion, and send the movement commands over the network to the Emotion Engine.

```bash
python3 controller.py
```
**Integrating controller.py via Network**

To ensure the AI inference pipeline (controller.py) can communicate seamlessly with the Procedural Animation Engine (emotions_g1_*.py) regardless of where each script is running (e.g., Inference on a powerful laptop, IK Engine on the robot), we use a standard UDP Socket.

In your controller.py, locate the function where the final emotion string (e.g., "HAPPY", "SAD") is decided. Replace the direct function calls with a simple UDP broadcast targeted at Port 5005.

Python Code Snippet for controller.py:

```python
import socket

def send_emotion_to_robot(emotion_string, target_ip="127.0.0.1"):
    """
    Sends the predicted emotion to the IK Engine via UDP.
    - target_ip: Use "127.0.0.1" if both scripts run on the same PC.
                 Use the Robot's IP (e.g., "192.168.1.126") if the IK Engine runs on the G1
                 and the inference runs on a remote laptop.
    """
    UDP_PORT = 5005
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Send the string encoded as bytes
        sock.sendto(emotion_string.encode('utf-8'), (target_ip, UDP_PORT))
        sock.close()
        print(f"[Network] Emotion '{emotion_string}' successfully sent to {target_ip}:{UDP_PORT}")
    except Exception as e:
        print(f"[Network Error] Could not send emotion: {e}")

# Example Usage inside your inference loop:
# predicted_emotion = model.predict(audio_feat, visual_feat)
# send_emotion_to_robot(predicted_emotion, target_ip="192.168.1.126")
```

Because the IK Engine is configured with SO_REUSEADDR and binds to 0.0.0.0, it will automatically intercept these network packets and trigger the corresponding procedural animation instantly.
