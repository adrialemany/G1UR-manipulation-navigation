import zmq
import cv2
import numpy as np
import time
import collections
import torch
import socket
from PIL import Image
import random

from insightface.app import FaceAnalysis
from inference import EmotionInference

ROBOT_IP = "192.168.0.107"

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
# 🔥 Robot Speech Map - Eng.ver
# ===============================
EMOTION_SPEECH_ENG_MAP = {

    "HAPPY": [  # 🙌 arms up
        "That makes me really happy!",
        "I'm so glad to hear that!",
        "That's wonderful news!",
        "I love that!",
        "You look really happy!",
        "That sounds amazing!"
    ],

    "DANCE": [  # 🎉 energetic / dynamic
        "Wow, that's exciting!",
        "I can feel your energy!",
        "This is really exciting!",
        "Let's celebrate that!",
        "That sounds like a lot of fun!",
        "I love this excitement!"
    ],

    "NEUTRAL": [  # 🤲 open hands
        "I see.",
        "That makes sense.",
        "Alright, I'm following you.",
        "I understand.",
        "Okay, go on.",
        "Hmm, I see what you mean."
    ],

    "FRUSTRATED": [  # 🤯 hands to head (confusion/stress)
        "That sounds really frustrating.",
        "I can see why that would be difficult.",
        "That must be stressful.",
        "I understand how that could be annoying.",
        "That’s not easy to deal with.",
        "Breath, everything has a solution."
    ],

    "SAD": [  # 😞 hands near face / low energy
        "I'm really sorry to hear that.",
        "That sounds tough.",
        "I'm here with you.",
        "Take your time.",
        "That must have been hard.",
        "I understand... that sounds difficult."
    ],

    "ANGRY": [  # 👊 forward / aggressive motion
        "Yeah, that’s really frustrating!",
        "That’s so annoying!",
        "Ugh, that’s really upsetting!",
        "I don’t like that either!",
        "That’s seriously not okay!",
        "Wow, that would make me mad too!"
    ]
}

# ===============================
# 🔥 Robot Speech Map - Spain.ver
# ===============================
EMOTION_SPEECH_SPA_MAP = {

    "HAPPY": [  # 🙌
        "¡Eso es maravilloso!",
        "¡Me alegra mucho escucharlo!",
        "¡Qué buena noticia!",
        "¡Me encanta eso!",
        "¡Te ves muy feliz!",
        "¡Suena genial!"
    ],

    "DANCE": [  # 🎉
        "¡Wow, eso suena emocionante!",
        "¡Siento tu alegría!",
        "¡Esto es muy emocionante!",
        "¡Vamos a celebrarlo!",
        "¡Suena muy divertido!",
        "¡Me encanta esta emoción!"
    ],

    "NEUTRAL": [  # 🤲
        "Ya veo.",
        "Entiendo.",
        "Vale, te sigo.",
        "Tiene sentido.",
        "De acuerdo, continúa.",
        "Hmm, entiendo lo que dices."
    ],

    "FRUSTRATED": [  # 🤯
        "Eso suena frustrante.",
        "Entiendo por qué sería difícil.",
        "Debe ser estresante.",
        "Puedo imaginar que eso molesta.",
        "No es fácil lidiar con eso.",
        "Respira hondo, de todo se sale."
    ],

    "SAD": [  # 😞
        "Lo siento mucho.",
        "Eso suena difícil.",
        "Estoy aquí contigo.",
        "Tómate tu tiempo.",
        "Debe haber sido duro.",
        "Entiendo... suena complicado."
    ],

    "ANGRY": [  # 👊
    "¡Eso es súper molesto!",
    "¡Qué rabia da eso!",
    "¡Eso es muy frustrante!",
    "¡No está nada bien!",
    "¡Uf, eso sí que enfada!",
    "¡A mí también me molestaría!"
    ]
}

# ===============================
# 🔥 Robot Questions - Eng.ver
# ===============================
QUESTIONS_ENG = [

    # HAPPY
    "Can you tell me about a recent moment that made you smile without realizing it?",

    # ANGRY
    "Was there a situation recently where something didn’t go the way you expected?",

    # EXCITED
    "Is there anything coming up that you're really looking forward to these days?",

    # SAD
    "Have you had a moment lately that stayed on your mind longer than you expected?",

    # FRUSTRATED
    "Have you been working on something recently that didn’t go as smoothly as you hoped?",

    # NEUTRAL
    "How has your day been so far?"
    
    # freedom
    "Is there anything you'd like to share about your day?"
]

# ===============================
# 🔥 Robot Questions - Spain.ver
# ===============================
QUESTIONS_SPA = [

    # HAPPY
    "¿Puedes contarme algún momento reciente en el que sonreíste sin darte cuenta?",

    # ANGRY
    "¿Ha habido alguna situación recientemente en la que algo no salió como esperabas?",

    # EXCITED
    "¿Hay algo próximamente que estés esperando con muchas ganas?",

    # SAD
    "¿Has tenido algún momento últimamente que se te haya quedado en la mente más de lo que esperabas?",

    # FRUSTRATED
    "¿Has estado trabajando en algo recientemente que no salió tan bien como esperabas?",

    # NEUTRAL
    "¿Cómo ha ido tu día hasta ahora?",

    # freedom
    "¿Hay algo que te gustaría compartir sobre tu día?"
]

def send_emotion_udp(emotion_str):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(emotion_str.encode('utf-8'), (ROBOT_IP, 5005))
    except Exception as e:
        print("❌ UDP send error:", e)

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
audio_buffer = collections.deque(maxlen=16000*6)

def slice_by_time(buffer, t_start, t_end):
    return [x for (t, x) in buffer if t_start <= t <= t_end]


# ======================
# 🔥 Robot Command
# ======================        
def send_cmd(cmd):
    print(f"🚀 SEND CMD: {cmd}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            s.connect((ROBOT_IP, 6000))
            s.sendall(cmd.encode('utf-8'))
            try:
                s.recv(1024)
                print("✅ robot responded")
            except:
                print("⚠️ no response from robot")
                pass
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
    
    # ======================
    # 🔥 INTRO
    # ======================
    state = "intro"
    question_idx = 0
    
    frame_buffer.clear()
    audio_buffer.clear()
    infer.frame_buffer.clear()
    
    # intro_text_eng = "Hi! Nice to meet you. Today, I’d like to talk with you about your recent experiences and feelings. There are no right or wrong answers, so feel free to speak comfortably. Let’s start!"
    intro_text_spa = "Hola, ¡encantado de conocerte! Hoy me gustaría hablar contigo sobre tus experiencias y sentimientos recientes. No hay respuestas correctas o incorrectas, así que siéntete libre de hablar con tranquilidad. ¡Empecemos!"

    time.sleep(3.0)

    # print("🔥 INTRO START")
    # send_cmd(f"speak:{intro_text_spa}")
    # time.sleep(len(intro_text_spa) * 0.09 + 4)

    # state = "ask_question"
    
    # ======================
    # 🔥 QUESTION LOOP
    # ======================
    camera_active = False
    human_start = None
    human_end = None
    last_audio_time = time.time()
    frame_count = 0
    last_print_time = 0
    current_emotion = None
    use_real_inference = False
    is_speaking = False

    while True:
        loop_start = time.time()

        # ======================
        # 🔊 AUDIO RECEIVE
        # ======================
        try:
            if is_speaking:
                raise zmq.Again()
        
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
        # 🔥 STATE MACHINE
        # ======================
        print("🔁 Current state:", state)

        if state == "intro":
            print("🔥 INTRO")

            is_speaking = True
            send_cmd(f"speak:{intro_text_spa}")
            time.sleep(len(intro_text_spa) * 0.09 + 4)
            is_speaking = False
        
            frame_buffer.clear()
            audio_buffer.clear()
            infer.frame_buffer.clear()
        
            last_audio_time = time.time()
        
            state = "ask_question"
            time.sleep(1.5)
            print("🔥 INTRO SENT")
            continue
        
        elif state == "ask_question":
            if question_idx >= len(QUESTIONS_SPA):
                send_cmd("speak:Ha sido un placer hablar contigo hoy. ¡Muchas gracias!")
                break

            use_real_inference = (question_idx == len(QUESTIONS_SPA) - 1)

            question = QUESTIONS_SPA[question_idx]
            print(f"🗣️ Question: {question}")

            speech_time = len(question) * 0.065

            is_speaking = True
            send_cmd(f"speak:{question}")
            time.sleep(speech_time + 1.0)

            # 🔥 residual audio 제거
            audio_buffer.clear()
            frame_buffer.clear()
            infer.frame_buffer.clear()
            
            last_audio_time = time.time()
            
            is_speaking = False
            
            human_start = None
            last_audio_time = time.time()
            
            camera_active = True
            state = "collecting"
            
            # 🔥 Decide inference Option
            # if question_idx == len(QUESTIONS_SPA) - 1:
            #     use_real_inference = True
                
            #     frame_buffer.clear()
            #     audio_buffer.clear()
            #     infer.frame_buffer.clear()
                
            #     human_start = None
            #     is_speaking = True
            #     # time.sleep(len(question) * 0.07 + 1.5)
            #     speech_time = len(QUESTIONS_SPA[question_idx]) * 0.065

            #     is_speaking = False
            #     time.sleep(speech_time + 1.0)
                
            #     frame_buffer.clear()
            #     audio_buffer.clear()
            #     infer.frame_buffer.clear()
            #     last_audio_time = time.time()

            #     camera_active = True
            #     state = "collecting"
                
            # else:
            #     use_real_inference = False

            #     frame_buffer.clear()
            #     audio_buffer.clear()
            #     infer.frame_buffer.clear()
            
            #     human_start = None
            
            #     speech_time = len(question) * 0.065
            #     time.sleep(speech_time + 1.0)

            #     frame_buffer.clear()
            #     audio_buffer.clear()
            #     infer.frame_buffer.clear()
            #     last_audio_time = time.time()
            
            #     camera_active = True
            #     state = "collecting"
                
            #     # PREDEFINED = ["HAPPY", "ANGRY", "DANCE", "SAD", "FRUSTRATED", "NEUTRAL"]
            #     # current_emotion = PREDEFINED[question_idx]

            #     # speech_time = len(question) * 0.065
            #     # time.sleep(speech_time + 1.0)

            #     # state = "react"
            #     # continue
            
        elif state == "next_question":
            question_idx += 1

            camera_active = False
            time.sleep(2)
            state = "ask_question"

        # ======================
        # 🎥 VIDEO RECEIVE
        # ======================
        if state == "collecting" and camera_active:
            try:
                raw = video_socket.recv(zmq.NOBLOCK)
                frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                now = time.time()
                
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

                # if now - human_start > 3.0 and len(frame_buffer) > 10:
                #     human_end = now
                #     state = "predict"
                    
                min_listen_time = 3.0
                max_wait_time = 20.0
                valid_face_frames = len(frame_buffer)

                if ((now - human_start > min_listen_time) and
                    (now - last_audio_time > 3.0) and 
                    valid_face_frames > 15) or \
                    (now - human_start > max_wait_time):
                    human_end = now
                    state = "predict"

            except zmq.Again:
                pass
            
        # ======================
        # 🔥 PREDICT
        # ======================
        elif state == "predict":
            
            if not use_real_inference:
                PREDEFINED = ["HAPPY", "ANGRY", "DANCE", "SAD", "FRUSTRATED", "NEUTRAL"]
                current_emotion = PREDEFINED[question_idx]
                
                state = "react"
                continue

            frames = slice_by_time(frame_buffer, human_start, human_end)
            audio_samples  = slice_by_time(audio_buffer, human_start, human_end)
            max_len = 16000 * 6
            
            if len(frames) < 8:
                current_emotion = "NEUTRAL"
                state = "react"
                continue
                # state = "ask_question"
                # continue

            frames = frames[-16:]

            processed = []
            for f in frames:
                img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                img = infer.transform(img)
                processed.append(img)

            frames_tensor = torch.stack(processed).unsqueeze(0).to(infer.device)
            
            if len(audio_samples) == 0:
                wav = np.zeros(16000 * 6)
            else:
                wav = np.array(audio_samples)

            if len(wav) < max_len:
                wav = np.pad(wav, (0, max_len - len(wav)))
            else:
                # wav = wav[:max_len]
                wav = wav[-16000*4:]

            wav_tensor = torch.tensor(wav).float().unsqueeze(0).to(infer.device)

            if use_real_inference:
                # 🔥 진짜 inference
                emotion = infer.predict(wav_tensor)
                emotion_str = EMOTION_MAP.get(emotion, "NEUTRAL")
            else:
                # 🔥 predefined
                PREDEFINED = ["HAPPY", "ANGRY", "DANCE", "SAD", "FRUSTRATED", "NEUTRAL"]
                emotion_str = PREDEFINED[question_idx]

            current_emotion = emotion_str
            state = "react"
    
            # emotion = infer.predict(wav_tensor)
            # emotion_str = EMOTION_MAP.get(emotion, "NEUTRAL")

            # current_emotion = emotion_str
            # state = "react" 
            continue

        elif state == "react":
            print("🟠 REACT")
        
            # 🔥 motion
            send_emotion_udp(current_emotion)
        
            # 👉 motion duration
            motion_duration = 3.0
            time.sleep(motion_duration)
        
            # 🔥 speech
            speech_list = EMOTION_SPEECH_SPA_MAP.get(current_emotion, ["Entiendo."])
            speech = random.choice(speech_list)

            is_speaking = True
            
            send_cmd(f"speak:{speech}")
        
            # 👉 wait during user's speech
            time.sleep(len(speech) * 0.07 + 1.5)

            is_speaking = False
        
            # 🔥 reset
            camera_active = False
            human_start = None
        
            frame_buffer.clear()
            audio_buffer.clear()
            infer.frame_buffer.clear()

            time.sleep(1.0)
            state = "next_question"

        # Check FPS
        fps = 1 / (time.time() - loop_start + 1e-6)
        if time.time() - last_print_time > 2:
            print(f"⏱ FPS: {fps:.2f}")
            last_print_time = time.time()
    
        time.sleep(0.01)

if __name__ == "__main__":
    main()

# import zmq
# import cv2
# import numpy as np
# import time
# import collections
# import torch
# import socket
# from PIL import Image
# import random

# from insightface.app import FaceAnalysis
# from inference import EmotionInference

# ROBOT_IP = "192.168.0.107"

# # ======================
# # 🔥 Face Detector
# # ======================
# face_app = FaceAnalysis(name='buffalo_l')
# face_app.prepare(ctx_id=0)

# EMOTION_MAP = {
#     0: "ANGRY",
#     1: "HAPPY",
#     2: "EXCITED", # dance?
#     3: "SAD",
#     4: "FRUSTRATED",
#     5: "NEUTRAL"
# }

# # ===============================
# # 🔥 Robot Speech Map - Eng.ver
# # ===============================
# EMOTION_SPEECH_ENG_MAP = {

#     "HAPPY": [  # 🙌 arms up
#         "That makes me really happy!",
#         "I'm so glad to hear that!",
#         "That's wonderful news!",
#         "I love that!",
#         "You look really happy!",
#         "That sounds amazing!"
#     ],

#     "EXCITED": [  # 🎉 energetic / dynamic
#         "Wow, that's exciting!",
#         "I can feel your energy!",
#         "This is really exciting!",
#         "Let's celebrate that!",
#         "That sounds like a lot of fun!",
#         "I love this excitement!"
#     ],

#     "NEUTRAL": [  # 🤲 open hands
#         "I see.",
#         "That makes sense.",
#         "Alright, I'm following you.",
#         "I understand.",
#         "Okay, go on.",
#         "Hmm, I see what you mean."
#     ],

#     "FRUSTRATED": [  # 🤯 hands to head (confusion/stress)
#         "That sounds really frustrating.",
#         "I can see why that would be difficult.",
#         "That must be stressful.",
#         "I understand how that could be annoying.",
#         "That’s not easy to deal with.",
#         "Breath, everything has a solution."
#     ],

#     "SAD": [  # 😞 hands near face / low energy
#         "I'm really sorry to hear that.",
#         "That sounds tough.",
#         "I'm here with you.",
#         "Take your time.",
#         "That must have been hard.",
#         "I understand... that sounds difficult."
#     ],

#     "ANGRY": [  # 👊 forward / aggressive motion
#         "Yeah, that’s really frustrating!",
#         "That’s so annoying!",
#         "Ugh, that’s really upsetting!",
#         "I don’t like that either!",
#         "That’s seriously not okay!",
#         "Wow, that would make me mad too!"
#     ]
# }

# # ===============================
# # 🔥 Robot Speech Map - Spain.ver
# # ===============================
# EMOTION_SPEECH_SPA_MAP = {

#     "HAPPY": [  # 🙌
#         "¡Eso es maravilloso!",
#         "¡Me alegra mucho escucharlo!",
#         "¡Qué buena noticia!",
#         "¡Me encanta eso!",
#         "¡Te ves muy feliz!",
#         "¡Suena genial!"
#     ],

#     "EXCITED": [  # 🎉
#         "¡Wow, eso suena emocionante!",
#         "¡Siento tu alegría!",
#         "¡Esto es muy emocionante!",
#         "¡Vamos a celebrarlo!",
#         "¡Suena muy divertido!",
#         "¡Me encanta esta emoción!"
#     ],

#     "NEUTRAL": [  # 🤲
#         "Ya veo.",
#         "Entiendo.",
#         "Vale, te sigo.",
#         "Tiene sentido.",
#         "De acuerdo, continúa.",
#         "Hmm, entiendo lo que dices."
#     ],

#     "FRUSTRATED": [  # 🤯
#         "Eso suena frustrante.",
#         "Entiendo por qué sería difícil.",
#         "Debe ser estresante.",
#         "Puedo imaginar que eso molesta.",
#         "No es fácil lidiar con eso.",
#         "Respira hondo, de todo se sale."
#     ],

#     "SAD": [  # 😞
#         "Lo siento mucho.",
#         "Eso suena difícil.",
#         "Estoy aquí contigo.",
#         "Tómate tu tiempo.",
#         "Debe haber sido duro.",
#         "Entiendo... suena complicado."
#     ],

#     "ANGRY": [  # 👊
#     "¡Eso es súper molesto!",
#     "¡Qué rabia da eso!",
#     "¡Eso es muy frustrante!",
#     "¡No está nada bien!",
#     "¡Uf, eso sí que enfada!",
#     "¡A mí también me molestaría!"
#     ]
# }

# # ===============================
# # 🔥 Robot Questions - Eng.ver
# # ===============================
# QUESTIONS_ENG = [

#     # HAPPY
#     "Can you tell me about a recent moment that made you smile without realizing it?",

#     # ANGRY
#     "Was there a situation recently where something didn’t go the way you expected?",

#     # EXCITED
#     "Is there anything coming up that you're really looking forward to these days?",

#     # SAD
#     "Have you had a moment lately that stayed on your mind longer than you expected?",

#     # FRUSTRATED
#     "Have you been working on something recently that didn’t go as smoothly as you hoped?",

#     # NEUTRAL
#     "How has your day been so far?"
    
#     # freedom
#     "Is there anything you'd like to share about your day?"
# ]

# # ===============================
# # 🔥 Robot Questions - Spain.ver
# # ===============================
# QUESTIONS_SPA = [

#     # HAPPY
#     "¿Puedes contarme algún momento reciente en el que sonreíste sin darte cuenta?",

#     # ANGRY
#     "¿Ha habido alguna situación recientemente en la que algo no salió como esperabas?",

#     # EXCITED
#     "¿Hay algo próximamente que estés esperando con muchas ganas?",

#     # SAD
#     "¿Has tenido algún momento últimamente que se te haya quedado en la mente más de lo que esperabas?",

#     # FRUSTRATED
#     "¿Has estado trabajando en algo recientemente que no salió tan bien como esperabas?",

#     # NEUTRAL
#     "¿Cómo ha ido tu día hasta ahora?",

#     # freedom
#     "¿Hay algo que te gustaría compartir sobre tu día?"
# ]

# def send_emotion_udp(emotion_str):
#     try:
#         sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         sock.sendto(emotion_str.encode('utf-8'), (ROBOT_IP, 5005))
#     except Exception as e:
#         print("❌ UDP send error:", e)

# def crop_face(frame):
#     faces = face_app.get(frame)

#     if len(faces) == 0:
#         print("❌ No face detected")
#         return None

#     areas = [(f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]) for f in faces]
#     face = faces[np.argmax(areas)]

#     x1, y1, x2, y2 = face.bbox.astype(int)

#     H, W = frame.shape[:2]
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(W, x2), min(H, y2)

#     crop = frame[y1:y2, x1:x2]

#     if crop.size == 0:
#         print("❌ Empty crop")
#         return None

#     crop = cv2.resize(crop, (224,224))
#     print("✅ Face detected")
#     return crop


# # ======================
# # 🔥 Buffers
# # ======================
# frame_buffer = collections.deque(maxlen=300)
# audio_buffer = collections.deque(maxlen=16000*10)

# def slice_by_time(buffer, t_start, t_end):
#     return [x for (t, x) in buffer if t_start <= t <= t_end]


# # ======================
# # 🔥 Robot Command
# # ======================
# def send_cmd(cmd):
#     print(f"🚀 SEND CMD: {cmd}")
#     try:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.connect((ROBOT_IP, 6000))
#             s.sendall(cmd.encode('utf-8'))
#             s.recv(1024)
#     except Exception as e:
#         print("❌ CMD send error:", e)


# # ======================
# # 🔥 main
# # ======================
# def main():

#     context = zmq.Context()

#     # video socket
#     video_socket = context.socket(zmq.SUB)
#     video_socket.setsockopt(zmq.CONFLATE, 1)
#     video_socket.setsockopt_string(zmq.SUBSCRIBE, "")
#     video_socket.connect(f"tcp://{ROBOT_IP}:6002")

#     # audio socket
#     audio_socket = context.socket(zmq.SUB)
#     audio_socket.setsockopt(zmq.CONFLATE, 1)
#     audio_socket.setsockopt_string(zmq.SUBSCRIBE, "")
#     audio_socket.connect(f"tcp://{ROBOT_IP}:6003")

#     print("✅ Connected to G1 video & audio")

#     infer = EmotionInference("best_epoch17_val0.5943.pth")
    
#     # ======================
#     # 🔥 INTRO
#     # ======================
#     state = "intro"
#     question_idx = 0
    
#     intro_text_eng = "Hi! Nice to meet you. Today, I’d like to talk with you about your recent experiences and feelings. There are no right or wrong answers, so feel free to speak comfortably. Let’s start!"
#     intro_text_spa = "Hola, ¡encantado de conocerte! Hoy me gustaría hablar contigo sobre tus experiencias y sentimientos recientes. No hay respuestas correctas o incorrectas, así que siéntete libre de hablar con tranquilidad. ¡Empecemos!"

#     # send_cmd(f"speak:{intro_text_spa}")
#     # time.sleep(len(intro_text_spa) * 0.09 + 6)
    
#     # ======================
#     # 🔥 QUESTION LOOP
#     # ======================
#     # state = "ask_question"
#     # question_idx = 0
#     camera_active = False
#     human_start = None
#     human_end = None

#     # state = "idle"
#     silence_threshold = 1.0

#     last_audio_time = 0

#     frame_count = 0

#     last_print_time = 0

#     current_emotion = None

#     while True:
#         loop_start = time.time()

#         # ======================
#         # 🔊 AUDIO RECEIVE
#         # ======================
#         try:
#             raw_audio = audio_socket.recv(zmq.NOBLOCK)
#             audio = np.frombuffer(raw_audio, dtype='float32')
#             audio = (audio * 32768).astype(np.int16)

#             now = time.time()

#             for sample in audio:
#                 audio_buffer.append((now, sample))

#             last_audio_time = now
#             print("🎤 Audio received:", len(audio))

#         except zmq.Again:
#             pass
        
#         # ======================
#         # 🔥 STATE MACHINE
#         # ======================
#         print("🔁 Current state:", state)

#         if state == "intro":
#             print("🟡 INTRO")
        
#             send_cmd(f"speak:{intro_text_spa}")
        
#             # 충분히 기다림 (중요)
#             time.sleep(len(intro_text_spa) * 0.09 + 4)

#             # 🔥 buffer reset (추천)
#             frame_buffer.clear()
#             audio_buffer.clear()
#             infer.frame_buffer.clear()
        
#             state = "ask_question"
#             continue
        
#         elif state == "ask_question":
#             if question_idx >= len(QUESTIONS_SPA):
#                 # send_cmd("speak:It was really nice talking with you today. Thank you!")
#                 send_cmd("speak:Ha sido un placer hablar contigo hoy. ¡Muchas gracias!")
#                 break

#             question = QUESTIONS_SPA[question_idx]
#             print(f"🗣️ Question: {question}")

#             send_cmd(f"speak:{question}")
#             frame_buffer.clear()
#             audio_buffer.clear()
#             infer.frame_buffer.clear()
        
#             human_start = None
    
#             time.sleep(max(4, len(question) * 0.06))
#             # time.sleep(len(question) * 0.09 + 4)

#             time.sleep(1.0)
#             camera_active = True
#             state = "collecting"
            
#         elif state == "next_question":
#             question_idx += 1

#             camera_active = False
#             time.sleep(2)
#             state = "ask_question"

#         # ======================
#         # 🎥 VIDEO RECEIVE
#         # ======================
#         if state == "collecting" and camera_active:
#             try:
#                 raw = video_socket.recv(zmq.NOBLOCK)
#                 frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)

#                 if frame is None:
#                     continue

#                 now = time.time()
                
#                 face = None
                
#                 if frame_count % 3 == 0:
#                     face = crop_face(frame)
#                 frame_count += 1

#                 if human_start is None:
#                     human_start = now
#                     infer.frame_buffer.clear()
                    
#                 if face is not None:
#                     frame_buffer.append((now, face))
#                     infer.update_frame(face)

#                 if now - human_start > 3.0 and len(frame_buffer) > 10:
#                     human_end = now
#                     state = "predict"
                
#                 # if now - last_audio_time > silence_threshold:
#                 #     human_end = now
#                 #     state = "predict"

#             except zmq.Again:
#                 pass
            
#         # ======================
#         # 🔥 PREDICT
#         # ======================
#         elif state == "predict":

#             frames = slice_by_time(frame_buffer, human_start, human_end)
#             audio_samples  = slice_by_time(audio_buffer, human_start, human_end)
#             max_len = 16000 * 6
            
#             if len(frames) < 8:
#                 state = "ask_question"
#                 continue

#             frames = frames[-16:]

#             processed = []
#             for f in frames:
#                 img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
#                 img = infer.transform(img)
#                 processed.append(img)

#             frames_tensor = torch.stack(processed).unsqueeze(0).to(infer.device)
            
#             if len(audio_samples) == 0:
#                 wav = np.zeros(16000 * 6)
#             else:
#                 wav = np.array(audio_samples)

#             if len(wav) < max_len:
#                 wav = np.pad(wav, (0, max_len - len(wav)))
#             else:
#                 wav = wav[:max_len]

#             wav_tensor = torch.tensor(wav).float().unsqueeze(0).to(infer.device)

#             emotion = infer.predict(wav_tensor)
#             emotion_str = EMOTION_MAP.get(emotion, "NEUTRAL")

#             current_emotion = emotion_str
#             state = "react" 
#             continue

#         elif state == "react":
#             print("🟠 REACT")
        
#             # 🔥 motion
#             send_emotion_udp(current_emotion)
        
#             # 👉 motion duration
#             motion_duration = 3.0
#             time.sleep(motion_duration)
        
#             # 🔥 speech
#             speech_list = EMOTION_SPEECH_SPA_MAP.get(current_emotion, ["Entiendo."])
#             speech = random.choice(speech_list)
        
#             send_cmd(f"speak:{speech}")
        
#             # 👉 speech 끝까지 기다림
#             time.sleep(len(speech) * 0.07 + 1.5)
        
#             # 🔥 reset
#             camera_active = False
#             human_start = None
        
#             frame_buffer.clear()
#             audio_buffer.clear()
#             infer.frame_buffer.clear()

#             time.sleep(1.0)
#             state = "next_question"

#             # print(f"🔥 Emotion: {emotion} → {emotion_str}")

#             # # 🔥 motion + LED
#             # send_emotion_udp(emotion_str)

#             # time.sleep(0.3)

#             # # 🔥 speech
#             # speech_list = EMOTION_SPEECH_SPA_MAP.get(emotion_str, ["I understand."])
#             # speech = random.choice(speech_list)

#             # send_cmd(f"speak:{speech}")

#             # # reset
#             # camera_active = False
#             # human_start = None
            
#             # frame_buffer.clear()
#             # audio_buffer.clear()
#             # infer.frame_buffer.clear()   # 🔥 이거 추가

#             # state = "next_question"

#         # Check FPS
#         fps = 1 / (time.time() - loop_start + 1e-6)
#         # print(f"⏱ FPS: {fps:.2f}")
#         if time.time() - last_print_time > 2:
#             print(f"⏱ FPS: {fps:.2f}")
#             last_print_time = time.time()
    
#         time.sleep(0.01)

# if __name__ == "__main__":
#     main()
