import zmq
import cv2
import numpy as np
import time
import collections
import socket
import random

from insightface.app import FaceAnalysis

ROBOT_IP = "192.168.0.107"

# ======================
# Face Detector
# ======================
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=0)

# ===============================
# Emotion Map
# ===============================
PREDEFINED_EMOTIONS = [
    "HAPPY",
    "ANGRY",
    "DANCE",
    "SAD",
    "FRUSTRATED",
    "NEUTRAL"
]

# ===============================
# Robot Speech Map - Spanish
# ===============================
EMOTION_SPEECH_SPA_MAP = {

    "HAPPY": [
        "¡Eso es maravilloso!",
        "¡Me alegra mucho escucharlo!",
        "¡Qué buena noticia!",
        "¡Me encanta eso!"
    ],

    "DANCE": [
        "¡Wow, eso suena emocionante!",
        "¡Vamos a celebrarlo!",
        "¡Suena muy divertido!"
    ],

    "NEUTRAL": [
        "Ya veo.",
        "Entiendo.",
        "Vale, te sigo."
    ],

    "FRUSTRATED": [
        "Eso suena frustrante.",
        "Debe ser estresante.",
        "No es fácil lidiar con eso."
    ],

    "SAD": [
        "Lo siento mucho.",
        "Eso suena difícil.",
        "Estoy aquí contigo."
    ],

    "ANGRY": [
        "¡Eso es súper molesto!",
        "¡Qué rabia da eso!",
        "¡Eso es muy frustrante!"
    ]
}

# ===============================
# Questions
# ===============================
QUESTIONS_SPA = [

    "¿Puedes contarme algún momento reciente en el que sonreíste sin darte cuenta?",

    "¿Ha habido alguna situación recientemente en la que algo no salió como esperabas?",

    "¿Hay algo próximamente que estés esperando con muchas ganas?",

    "¿Has tenido algún momento últimamente que se te haya quedado en la mente más de lo que esperabas?",

    "¿Has estado trabajando en algo recientemente que no salió tan bien como esperabas?",

    "¿Cómo ha ido tu día hasta ahora?"
]

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
# Wait Helper
# ===============================
def wait_seconds(sec):
    start = time.time()

    while time.time() - start < sec:
        time.sleep(0.01)

# ===============================
# MAIN
# ===============================
def main():

    intro_text = (
        "Hola, ¡encantado de conocerte! Hoy me gustaría hablar contigo sobre tus experiencias y sentimientos recientes. No hay respuestas correctas o incorrectas, así que siéntete libre de hablar con tranquilidad. ¡Empecemos!"
        # "Hola, encantado de conocerte. "
        # "Hoy me gustaría hablar contigo libremente."
    )

    print("START SCRIPTED INTERACTION")

    time.sleep(2)

    # ======================
    # INTRO
    # ======================
    send_cmd(f"speak:{intro_text}")

    intro_wait = len(intro_text) * 0.08 + 4.0
    wait_seconds(intro_wait)

    # ======================
    # QUESTION LOOP
    # ======================
    for idx, question in enumerate(QUESTIONS_SPA):

        emotion = PREDEFINED_EMOTIONS[idx]

        print("=" * 50)
        print(f"QUESTION {idx+1}")
        print("QUESTION:", question)
        print("EMOTION:", emotion)

        # ----------------------
        # Ask Question
        # ----------------------
        send_cmd(f"speak:{question}")

        question_wait = len(question) * 0.11 + 6.0
        wait_seconds(question_wait)

        # ----------------------
        # Emotion Reaction
        # ----------------------
        send_emotion_udp(emotion)

        time.sleep(2.5)

        speech = random.choice(
            EMOTION_SPEECH_SPA_MAP.get(emotion, ["Entiendo."])
        )

        send_cmd(f"speak:{speech}")

        speech_wait = len(speech) * 0.11 + 3.0
        wait_seconds(speech_wait)

        time.sleep(2.0)

    # ======================
    # END
    # ======================
    end_text = (
        "Ha sido un placer hablar contigo hoy. "
        "Muchas gracias."
    )

    send_cmd(f"speak:{end_text}")

    print("SCRIPTED INTERACTION FINISHED")


if __name__ == "__main__":
    main()