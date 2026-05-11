import zmq
import cv2
import numpy as np
import time
import collections
import torch
import socket
import random

from PIL import Image

from insightface.app import FaceAnalysis
from inference import EmotionInference

ROBOT_IP = "192.168.0.107"

# ======================
# Face Detector
# ======================
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

# ======================
# Emotion Map
# ======================
EMOTION_MAP = {
    0: "ANGRY",
    1: "HAPPY",
    2: "DANCE",
    3: "SAD",
    4: "FRUSTRATED",
    5: "NEUTRAL"
}

# ===============================
# Speech Map
# ===============================
EMOTION_SPEECH_SPA_MAP = {

    "HAPPY": [
        "¡Eso es maravilloso!",
        "¡Me alegra mucho escucharlo!"
    ],

    "DANCE": [
        "¡Wow, eso suena emocionante!",
        "¡Vamos a celebrarlo!"
    ],

    "NEUTRAL": [
        "Ya veo.",
        "Entiendo."
    ],

    "FRUSTRATED": [
        "Eso suena frustrante.",
        "Debe ser estresante."
    ],

    "SAD": [
        "Lo siento mucho.",
        "Estoy aquí contigo."
    ],

    "ANGRY": [
        "¡Eso es súper molesto!",
        "¡Qué rabia da eso!"
    ]
}

# ===============================
# Free Question
# ===============================
FREE_QUESTION = (
    "¿Hay algo que te gustaría compartir sobre tu día?"
)

# ===============================
# Buffers
# ===============================
frame_buffer = collections.deque(maxlen=300)
audio_buffer = collections.deque(maxlen=16000 * 8)

# ===============================
# UDP Emotion Send
# ===============================
def send_emotion_udp(emotion_str):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(emotion_str.encode('utf-8'), (ROBOT_IP, 5005))
    except Exception as e:
        print("UDP send error:", e)

# ===============================
# Robot Command
# ===============================
def send_cmd(cmd):

    print(f"SEND CMD: {cmd}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

            s.settimeout(1.0)
            s.connect((ROBOT_IP, 6000))

            msg = cmd + "\n"
            s.sendall(msg.encode('utf-8'))

            try:
                s.recv(1024)
            except:
                pass

    except Exception as e:
        print("CMD send error:", e)

# ===============================
# Face Crop
# ===============================
def crop_face(frame):

    faces = face_app.get(frame)

    if len(faces) == 0:
        return None

    areas = [
        (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])
        for f in faces
    ]

    face = faces[np.argmax(areas)]

    x1, y1, x2, y2 = face.bbox.astype(int)

    H, W = frame.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (224, 224))

    return crop

# ===============================
# Slice Helper
# ===============================
def slice_by_time(buffer, t_start, t_end):
    return [x for (t, x) in buffer if t_start <= t <= t_end]

# ===============================
# MAIN
# ===============================
def main():

    context = zmq.Context()

    # ----------------------
    # Video Socket
    # ----------------------
    video_socket = context.socket(zmq.SUB)

    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    video_socket.connect(f"tcp://{ROBOT_IP}:6002")

    # ----------------------
    # Audio Socket
    # ----------------------
    audio_socket = context.socket(zmq.SUB)

    audio_socket.setsockopt(zmq.CONFLATE, 1)
    audio_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    audio_socket.connect(f"tcp://{ROBOT_IP}:6003")

    print("CONNECTED")

    infer = EmotionInference("best_epoch17_val0.5943.pth")

    # ======================
    # INTRO
    # ======================
    intro_text = (
        "Hola, ¡encantado de conocerte! Hoy me gustaría hablar contigo sobre tus experiencias y sentimientos recientes. No hay respuestas correctas o incorrectas, así que siéntete libre de hablar con tranquilidad. ¡Empecemos!"
        # "Hola, encantado de conocerte. "
        # "Hoy me gustaría hablar contigo libremente."
    )

    send_cmd(f"speak:{intro_text}")

    time.sleep(len(intro_text) * 0.08 + 4.0)

    # ======================
    # ASK QUESTION
    # ======================
    send_cmd(f"speak:{FREE_QUESTION}")

    time.sleep(len(FREE_QUESTION) * 0.11 + 3.0)

    # ======================
    # LISTENING
    # ======================
    print("START LISTENING")

    human_start = None

    frame_count = 0

    while True:

        # ----------------------
        # AUDIO RECEIVE
        # ----------------------
        try:

            raw_audio = audio_socket.recv(zmq.NOBLOCK)

            audio = np.frombuffer(
                raw_audio,
                dtype='float32'
            )

            audio = (audio * 32768).astype(np.int16)

            now = time.time()

            for sample in audio:
                audio_buffer.append((now, sample))

        except zmq.Again:
            pass

        # ----------------------
        # VIDEO RECEIVE
        # ----------------------
        try:

            raw = video_socket.recv(zmq.NOBLOCK)

            frame = cv2.imdecode(
                np.frombuffer(raw, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                continue

            now = time.time()

            face = None

            if frame_count % 3 == 0:
                face = crop_face(frame)

            frame_count += 1

            if human_start is None:
                human_start = now

            if face is not None:
                frame_buffer.append((now, face))

            # ----------------------
            # Stop Condition
            # ----------------------
            if now - human_start > 8.0:
                human_end = now
                break

        except zmq.Again:
            pass

        time.sleep(0.01)

    # ======================
    # PREPARE INPUT
    # ======================
    frames = slice_by_time(
        frame_buffer,
        human_start,
        human_end
    )

    audio_samples = slice_by_time(
        audio_buffer,
        human_start,
        human_end
    )

    # ======================
    # FALLBACK
    # ======================
    if len(frames) < 8:

        emotion_str = "NEUTRAL"

    else:

        frames = frames[-16:]

        processed = []

        for f in frames:

            img = Image.fromarray(
                cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            )

            img = infer.transform(img)

            processed.append(img)

        frames_tensor = torch.stack(
            processed
        ).unsqueeze(0).to(infer.device)

        max_len = 16000 * 6

        if len(audio_samples) == 0:
            wav = np.zeros(max_len)
        else:
            wav = np.array(audio_samples)

        if len(wav) < max_len:
            wav = np.pad(
                wav,
                (0, max_len - len(wav))
            )
        else:
            wav = wav[-16000*4:]

        wav_tensor = torch.tensor(
            wav
        ).float().unsqueeze(0).to(infer.device)

        # ======================
        # INFERENCE
        # ======================
        emotion = infer.predict(wav_tensor)

        emotion_str = EMOTION_MAP.get(
            emotion,
            "NEUTRAL"
        )

    print("PREDICTED:", emotion_str)

    # ======================
    # REACTION
    # ======================
    send_emotion_udp(emotion_str)

    time.sleep(2.5)

    speech = random.choice(
        EMOTION_SPEECH_SPA_MAP.get(
            emotion_str,
            ["Entiendo."]
        )
    )

    send_cmd(f"speak:{speech}")

    time.sleep(len(speech) * 0.11 + 3.0)

    # ======================
    # END
    # ======================
    end_text = (
        "Muchas gracias por compartir conmigo."
    )

    send_cmd(f"speak:{end_text}")

    print("FREE INTERACTION FINISHED")


if __name__ == "__main__":
    main()