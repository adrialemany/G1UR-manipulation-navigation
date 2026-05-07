import zmq
import cv2
import numpy as np
import time
import collections
import torch
import socket
from PIL import Image

from insightface.app import FaceAnalysis
from inference import EmotionInference
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

ROBOT_IP = "192.168.0.107"

# ======================
# 🔥 Face Detector
# ======================
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

def crop_face(frame):
    faces = face_app.get(frame)

    if len(faces) == 0:
        print("❌ No face detected")
        return None

    areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
    face = faces[np.argmax(areas)]

    x1, y1, x2, y2 = face.bbox.astype(int)

    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        print("❌ Empty crop")
        return None

    crop = cv2.resize(crop, (224,224))
    print("✅ Face detected")
    return crop


# ======================
# 🔥 Buffers
# ======================
frame_buffer = collections.deque(maxlen=300)
audio_buffer = collections.deque(maxlen=16000*10)

def slice_by_time(buffer, t_start, t_end):
    return [x for (t, x) in buffer if t_start <= t <= t_end]


# ======================
# 🔥 Robot Command
# ======================
def send_cmd(cmd):
    print(f"🚀 SEND CMD: {cmd}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ROBOT_IP, 6000))
            s.sendall(cmd.encode('utf-8'))
            s.recv(1024)
    except Exception as e:
        print("❌ CMD send error:", e)


# ======================
# 🔥 main
# ======================
def main():

    context = zmq.Context()

    # video socket
    video_socket = context.socket(zmq.SUB)
    video_socket.setsockopt(zmq.CONFLATE, 1)
    video_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    video_socket.connect(f"tcp://{ROBOT_IP}:6002")

    # audio socket
    audio_socket = context.socket(zmq.SUB)
    audio_socket.setsockopt(zmq.CONFLATE, 1)
    audio_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    audio_socket.connect(f"tcp://{ROBOT_IP}:6003")

    print("✅ Connected to G1 video & audio")

    infer = EmotionInference("best_epoch17_val0.5943.pth")

    state = "idle"
    silence_threshold = 1.0

    last_audio_time = 0
    human_start = None

    frame_count = 0

    last_print_time = 0

    ChannelFactoryInitialize(0, "eth0")

    arm = G1ArmActionClient(); arm.Init()

    while True:

        loop_start = time.time()

        # ======================
        # 🔊 AUDIO RECEIVE
        # ======================
        try:
            raw_audio = audio_socket.recv(zmq.NOBLOCK)
            audio = np.frombuffer(raw_audio, dtype='float32')
            audio = (audio * 32768).astype(np.int16)

            now = time.time()

            for sample in audio:
                audio_buffer.append((now, sample))

            last_audio_time = now
            print("🎤 Audio received:", len(audio))

        except zmq.Again:
            pass

        # ======================
        # 🎥 VIDEO RECEIVE
        # ======================
        try:
            raw = video_socket.recv(zmq.NOBLOCK)
            frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

            if frame is None:
                continue

            print("📷 Frame received")

            # DEBUG: 화면 출력
            cv2.imwrite("debug.jpg", frame)

            now = time.time()

            # face crop
            if frame_count % 3 == 0:
                face = crop_face(frame)
            frame_count += 1

            if face is None:
                continue

            frame_buffer.append((now, face))

            # ======================
            # 🔥 STATE MACHINE
            # ======================
            print("🔁 Current state:", state)

            if state == "idle":
                print("🟡 IDLE → collecting")
                human_start = now
                infer.frame_buffer.clear()
                state = "collecting"

            elif state == "collecting":

                print("🔵 Collecting...")
                infer.update_frame(face)
            
                # 🔥 시간 기반 trigger
                if now - human_start > 3.0:
                    print("⏱ Force predict (3 sec)")
                    human_end = now
                    state = "predict"

            # elif state == "collecting":
            #     print("🔵 Collecting...")
            #     infer.update_frame(face)

                # if now - last_audio_time > silence_threshold:
                #     print("🔇 Silence detected → predict")
                #     human_end = now
                #     state = "predict"

            elif state == "predict":

                print("🟢 Predicting...")

                frames = slice_by_time(frame_buffer, human_start, human_end)
                audio  = slice_by_time(audio_buffer, human_start, human_end)

                print(f"Frames: {len(frames)}, Audio: {len(audio)}")

                if len(frames) < 8:
                    print("❌ Not enough frames")
                    state = "idle"
                    continue

                frames = frames[-16:]

                processed = []
                for f in frames:
                    img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                    img = infer.transform(img)
                    processed.append(img)

                frames_tensor = torch.stack(processed).unsqueeze(0).to(infer.device)

                wav = np.array(audio)
                max_len = 16000 * 6

                if len(wav) < max_len:
                    wav = np.pad(wav, (0, max_len - len(wav)))
                else:
                    wav = wav[:max_len]

                wav_tensor = torch.tensor(wav).float().unsqueeze(0).to(infer.device)

                emotion = infer.predict(wav_tensor)
                # emotion = infer.predict(wav_tensor, frames_tensor)

                print("🔥 Emotion:", emotion)

                # ======================
                # 🔥 Gesture Mapping
                # ======================
                if emotion == 0:  # happy
                    arm.ExecuteAction(17)
                    # send_cmd("stand")
                    # send_cmd("1")

                elif emotion == 1:
                    # send_cmd("squat")
                    arm.ExecuteAction(27)

                elif emotion == 2:
                    # send_cmd("0")
                    arm.ExecuteAction(27)

                elif emotion == 3:
                    # send_cmd("3")
                    arm.ExecuteAction(26)

                elif emotion == 4:
                    # send_cmd("stop")
                    arm.ExecuteAction(17)

                elif emotion == 5:
                    # send_cmd("speak:jumpscare")
                    arm.ExecuteAction(17)

                state = "idle"

        except zmq.Again:
            pass

        # FPS 확인
        fps = 1 / (time.time() - loop_start + 1e-6)
        # print(f"⏱ FPS: {fps:.2f}")
        if time.time() - last_print_time > 2:
            print(f"⏱ FPS: {fps:.2f}")
            last_print_time = time.time()
    
        time.sleep(0.01)


if __name__ == "__main__":
    main()
