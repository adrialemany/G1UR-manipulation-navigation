import cv2
import numpy as np
import time
import collections
import torch
import socket
import threading
import random
import sounddevice as sd
import sys
from PIL import Image

from insightface.app import FaceAnalysis
from inference import EmotionInference

# --- PARA SIMULACIÓN, TODO APUNTA A LOCALHOST ---
ROBOT_IP = "127.0.0.1" 

# ======================
# 🔥 Face Detector
# ======================
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

EMOTION_MAP = {
    0: "ANGRY",
    1: "HAPPY",
    2: "DANCE", # excited
    3: "SAD",
    4: "FRUSTRATED",
    5: "NEUTRAL"
}

# ===============================
# 🔥 Robot Speech Map - Spain.ver
# ===============================
EMOTION_SPEECH_SPA_MAP = {
    "HAPPY": [  # 🙌
        "¡Eso es maravilloso!", "¡Me alegra mucho escucharlo!",
        "¡Qué buena noticia!", "¡Me encanta eso!",
        "¡Te ves muy feliz!", "¡Suena genial!"
    ],
    "DANCE": [  # 🎉
        "¡Wow, eso suena emocionante!", "¡Siento tu alegría!",
        "¡Esto es muy emocionante!", "¡Vamos a celebrarlo!",
        "¡Suena muy divertido!", "¡Me encanta esta emoción!"
    ],
    "NEUTRAL": [  # 🤲
        "Ya veo.", "Entiendo.", "Vale, te sigo.",
        "Tiene sentido.", "De acuerdo, continúa.", "Hmm, entiendo lo que dices."
    ],
    "FRUSTRATED": [  # 🤯
        "Eso suena frustrante.", "Entiendo por qué sería difícil.",
        "Debe ser estresante.", "Puedo imaginar que eso molesta.",
        "No es fácil lidiar con eso.", "Respira hondo, de todo se sale."
    ],
    "SAD": [  # 😞
        "Lo siento mucho.", "Eso suena difícil.", "Estoy aquí contigo.",
        "Tómate tu tiempo.", "Debe haber sido duro.", "Entiendo... suena complicado."
    ],
    "ANGRY": [  # 👊
        "¡Eso es súper molesto!", "¡Qué rabia da eso!",
        "¡Eso es muy frustrante!", "¡No está nada bien!",
        "¡Uf, eso sí que enfada!", "¡A mí también me molestaría!"
    ]
}

QUESTIONS_SPA = [
    "¿Puedes contarme algún momento reciente en el que sonreíste sin darte cuenta?",
    "¿Ha habido alguna situación recientemente en la que algo no salió como esperabas?",
    "¿Hay algo próximamente que estés esperando con muchas ganas?",
    "¿Has tenido algún momento últimamente que se te haya quedado en la mente más de lo que esperabas?",
    "¿Has estado trabajando en algo recientemente que no salió tan bien como esperabas?",
    "¿Cómo ha ido tu día hasta ahora?",
    "¿Hay algo que te gustaría compartir sobre tu día?"
]

# ======================
# 🔥 Buffers & Audio Local
# ======================
frame_buffer = collections.deque(maxlen=300)
audio_buffer = collections.deque(maxlen=16000*6)
audio_lock = threading.Lock()
last_audio_time = 0

def audio_callback(indata, frames, time_info, status):
    global last_audio_time
    now = time.time()
    
    # Detector de volumen (VAD simple) para saber cuándo te callas
    volume = np.linalg.norm(indata[:, 0])
    if volume > 0.02:  # Umbral arbitrario de volumen para ruido de fondo
        last_audio_time = now

    with audio_lock:
        for sample in indata[:, 0]:
            audio_buffer.append((now, sample * 32768))

def slice_by_time(buffer, t_start, t_end, is_audio=False):
    if is_audio:
        with audio_lock:
            return [x for (t, x) in buffer if t_start <= t <= t_end]
    else:
        return [x for (t, x) in buffer if t_start <= t <= t_end]

# ======================
# 🔥 UI & Helpers
# ======================
def smart_sleep(duration, cap, state_text, emotion_text=""):
    """
    Duerme el programa el tiempo necesario, pero mantiene viva la ventana 
    de OpenCV actualizando los frames de la cámara para que no se congele.
    """
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret:
            display = frame.copy()
            cv2.putText(display, f"State: {state_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if emotion_text:
                cv2.putText(display, f"Emotion: {emotion_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            cv2.imshow("Vision del Robot", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)

def send_emotion_udp(emotion_str):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(emotion_str.encode('utf-8'), (ROBOT_IP, 5005))
        sock.close()
    except Exception as e:
        print("❌ UDP send error:", e)

def send_cmd(cmd):
    print(f"🚀 SEND CMD: {cmd}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect((ROBOT_IP, 6000))
            s.sendall(cmd.encode('utf-8'))
            try: s.recv(1024)
            except: pass
    except Exception as e:
        print("❌ CMD send error:", e)

def crop_face(frame):
    faces = face_app.get(frame)
    if len(faces) == 0:
        return None

    areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
    face = faces[np.argmax(areas)]

    x1, y1, x2, y2 = face.bbox.astype(int)
    H, W = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (224,224))

# ======================
# 🔥 main
# ======================
def main():
    print("✅ Iniciando cámara y micrófono locales para Simulación...")

    # Activar cámara y micro de tu portátil
    cap = cv2.VideoCapture(0)
    audio_stream = sd.InputStream(samplerate=16000, channels=1, callback=audio_callback)
    audio_stream.start()

    cv2.namedWindow("Vision del Robot")

    infer = EmotionInference("best_epoch17_val0.5943.pth")
    
    # ======================
    # 🔥 INTRO
    # ======================
    state = "intro"
    question_idx = 0
    current_emotion = None
    
    frame_buffer.clear()
    audio_buffer.clear()
    infer.frame_buffer.clear()
    
    intro_text_spa = "Hola, ¡encantado de conocerte! Hoy me gustaría hablar contigo sobre tus experiencias y sentimientos recientes. No hay respuestas correctas o incorrectas, así que siéntete libre de hablar con tranquilidad. ¡Empecemos!"

    smart_sleep(2.0, cap, state)
    send_cmd("ping")
    smart_sleep(0.5, cap, state)

    print("🔥 INTRO START")
    send_cmd(f"speak:{intro_text_spa}")
    smart_sleep(len(intro_text_spa) * 0.09 + 4, cap, state)

    state = "ask_question"
    
    # Variables de control
    camera_active = False
    human_start = None
    human_end = None
    frame_count = 0
    last_print_time = 0
    use_real_inference = False

    while True:
        loop_start = time.time()
        now = time.time()

        # Extraemos el frame en cada vuelta 
        ret, frame = cap.read()

        # ======================
        # 🔥 STATE MACHINE
        # ======================
        if state == "ask_question":
            if question_idx >= len(QUESTIONS_SPA):
                send_cmd("speak:Ha sido un placer hablar contigo hoy. ¡Muchas gracias!")
                break

            question = QUESTIONS_SPA[question_idx]
            print(f"🗣️ Question: {question}")
            send_cmd(f"speak:{question}")
            
            # 🔥 Decide inference Option
            if question_idx == len(QUESTIONS_SPA) - 1:
                use_real_inference = True
                frame_buffer.clear()
                audio_buffer.clear()
                infer.frame_buffer.clear()
                human_start = None
                
                speech_time = len(QUESTIONS_SPA[question_idx]) * 0.065
                smart_sleep(speech_time + 1.0, cap, state)

                camera_active = True
                state = "collecting"
                
            else:
                use_real_inference = False

                PREDEFINED = ["HAPPY", "ANGRY", "DANCE", "SAD", "FRUSTRATED", "NEUTRAL"]
                current_emotion = PREDEFINED[question_idx]
 
                frame_buffer.clear()
                audio_buffer.clear()
                infer.frame_buffer.clear()

                human_start = None
 
                speech_time = len(question) * 0.065
                time.sleep(speech_time + 2.0)

                camera_active = True
                state = "collecting"
                continue
            
        elif state == "next_question":
            question_idx += 1
            camera_active = False
            smart_sleep(2.0, cap, state)
            state = "ask_question"

        # ======================
        # 🎥 VIDEO PROCESS
        # ======================
        if state == "collecting" and camera_active and ret:
            face = None
            if frame_count % 3 == 0:
                face = crop_face(frame)
            frame_count += 1

            if human_start is None:
                human_start = now
                infer.frame_buffer.clear()
                
            if face is not None:
                frame_buffer.append((now, face))
                infer.update_frame(face)

            min_listen_time = 2.0
            max_wait_time = 10.0
            valid_face_frames = len(frame_buffer)

            # Usamos last_audio_time calculado por el callback de volumen
            if ((now - human_start > min_listen_time) and
                (now - last_audio_time > 1.8) and 
                valid_face_frames > 15) or \
                (now - human_start > max_wait_time):
                
                human_end = now
                state = "predict"

        # ======================
        # 🔥 PREDICT
        # ======================
        elif state == "predict":
            if not use_real_inference:
                PREDEFINED = ["HAPPY", "ANGRY", "DANCE", "SAD", "FRUSTRATED", "NEUTRAL"]
                current_emotion = PREDEFINED[question_idx]
                state = "react"
                continue

            frames = slice_by_time(frame_buffer, human_start, human_end, is_audio=False)
            audio_samples  = slice_by_time(audio_buffer, human_start, human_end, is_audio=True)
            max_len = 16000 * 6
            
            if len(frames) < 8:
                state = "ask_question"
                continue

            frames = frames[-16:]
            processed = []
            for f in frames:
                img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                img = infer.transform(img)
                processed.append(img)

            frames_tensor = torch.stack(processed).unsqueeze(0).to(infer.device)
            
            if len(audio_samples) == 0:
                wav = np.zeros(max_len)
            else:
                wav = np.array(audio_samples)

            if len(wav) < max_len:
                wav = np.pad(wav, (0, max_len - len(wav)))
            else:
                wav = wav[-max_len:]

            wav_tensor = torch.tensor(wav).float().unsqueeze(0).to(infer.device)

            if use_real_inference:
                emotion = infer.predict(wav_tensor)
                emotion_str = EMOTION_MAP.get(emotion, "NEUTRAL")
            else:
                PREDEFINED = ["HAPPY", "ANGRY", "DANCE", "SAD", "FRUSTRATED", "NEUTRAL"]
                emotion_str = PREDEFINED[question_idx]

            current_emotion = emotion_str
            state = "react"
            continue

        # ======================
        # 🔥 REACT
        # ======================
        elif state == "react":
            print(f"🟠 REACT: {current_emotion}")
        
            # 🔥 motion
            send_emotion_udp(current_emotion)
        
            motion_duration = 3.0
            smart_sleep(motion_duration, cap, state, current_emotion)
        
            # 🔥 speech
            speech_list = EMOTION_SPEECH_SPA_MAP.get(current_emotion, ["Entiendo."])
            speech = random.choice(speech_list)
        
            send_cmd(f"speak:{speech}")
        
            smart_sleep(len(speech) * 0.07 + 1.5, cap, state, current_emotion)
        
            # 🔥 reset
            camera_active = False
            human_start = None
        
            frame_buffer.clear()
            audio_buffer.clear()
            infer.frame_buffer.clear()

            smart_sleep(1.0, cap, state)
            state = "next_question"

        # ======================
        # 📺 DIBUJAR VENTANA PRINCIPAL
        # ======================
        if ret:
            display = frame.copy()
            cv2.putText(display, f"State: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if state in ["react", "predict", "next_question"] and current_emotion:
                cv2.putText(display, f"Emotion: {current_emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            cv2.imshow("Vision del Robot", display)
            # Refrescar y escuchar salida 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        fps = 1 / (time.time() - loop_start + 1e-6)
        if time.time() - last_print_time > 2:
            print(f"⏱ FPS: {fps:.2f} | Estado: {state}")
            last_print_time = time.time()

if __name__ == "__main__":
    main()
