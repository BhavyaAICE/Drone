import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import os
from twilio.rest import Client
import threading

# ================= THREAD SAFETY =================
mode_lock = threading.Lock()

# ================= TWILIO =================
ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_NUMBER = os.getenv("TWILIO_NUMBER")
YOUR_NUMBER = os.getenv("YOUR_NUMBER")

twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN) if ACCOUNT_SID and AUTH_TOKEN else None
last_sms_time = 0

def send_sms(msg):
    global last_sms_time
    if not twilio_client:
        return
    if time.time() - last_sms_time < 60:
        return
    last_sms_time = time.time()
    twilio_client.messages.create(body=msg, from_=TWILIO_NUMBER, to=YOUR_NUMBER)

# ================= CONFIG =================
RUN_THRESHOLD = 25
HISTORY_LEN = 6
OVERCROWD_LIMIT = 8
LOCK_DURATION = 4

# ================= GLOBAL STATE =================
mode = None
previous_mode = None

# ================= FPS CONTROL (CRITICAL FIX) =================
_last_process_time = 0
MIN_FRAME_INTERVAL = 0.12   # ~8 FPS (stable for YOLO)

# ================= MODELS =================
yolo = YOLO("yolov8n.pt")

pose = mp.solutions.pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

tracker = DeepSort(max_age=30)

# ================= TARGET FACE =================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = None
LOCK_ID = None
lock_time = 0

if os.path.exists("SRK.jpg"):
    img = cv2.imread("SRK.jpg", cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = cv2.resize(img[y:y+h, x:x+w], (200, 200))
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train([face], np.array([1]))

# ================= STATE =================
track_history = {}

def reset_state():
    global track_history, LOCK_ID
    track_history.clear()
    LOCK_ID = None

# ================= CORE =================
def process_frame(frame):
    global previous_mode, LOCK_ID, lock_time, _last_process_time


    frame = cv2.resize(frame, (640, 360))

    with mode_lock:
        active_mode = mode

    # Always show mode
    cv2.putText(
        frame,
        f"MODE: {active_mode.upper() if active_mode else 'IDLE'}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # Reset when mode changes
    if active_mode != previous_mode:
        reset_state()
        previous_mode = active_mode

    # Idle → just show camera
    if active_mode is None:
        return frame

    results = yolo.predict(frame, conf=0.5, classes=[0], verbose=False)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ================= SOS MODE =================
    if active_mode == "sos":
        suspicious_count = 0

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                crop = rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                label = "SAFE"
                color = (0, 255, 0)

                pose_result = pose.process(crop)
                if pose_result.pose_landmarks:
                    lm = pose_result.pose_landmarks.landmark
                    if lm[15].y < lm[11].y or lm[16].y < lm[12].y:
                        suspicious_count += 1
                        label = "FATAL"
                        color = (0, 0, 255)

                cx, cy = (x1 + x2)//2, (y1 + y2)//2
                pid = (x1, y1, x2, y2)
                track_history.setdefault(pid, deque(maxlen=HISTORY_LEN)).append((cx, cy))

                if len(track_history[pid]) >= 2:
                    dx = track_history[pid][-1][0] - track_history[pid][0][0]
                    dy = track_history[pid][-1][1] - track_history[pid][0][1]
                    if np.hypot(dx, dy) > RUN_THRESHOLD:
                        suspicious_count += 1
                        label = "RUNNING"
                        color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if suspicious_count == 0:
            cv2.putText(frame, "NO SUSPICIOUS ACTIVITY",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

    # ================= CROWD MODE =================
    elif active_mode == "crowd":
        count = sum(len(r.boxes) for r in results)
        cv2.putText(frame, f"People: {count}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if count >= OVERCROWD_LIMIT:
            cv2.putText(frame, "OVERCROWD!", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            send_sms(f"Overcrowd detected: {count}")

    # ================= TARGET MODE =================
    elif active_mode == "target" and recognizer:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []

        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2 - x1, y2 - y1], 0.9, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)
        found = False

        for t in tracks:
            if not t.is_confirmed():
                continue

            tid = t.track_id
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            person = gray[y1:y2, x1:x2]

            faces = face_cascade.detectMultiScale(person, 1.3, 5)
            for (fx, fy, fw, fh) in faces:
                if fw < 80 or fh < 80:
                    continue

                face = cv2.resize(person[fy:fy+fh, fx:fx+fw], (200, 200))
                _, conf = recognizer.predict(face)
                if conf < 80:
                    LOCK_ID = tid
                    lock_time = time.time()
                    found = True

            if LOCK_ID == tid:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "TARGET LOCKED", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if LOCK_ID and not found and time.time() - lock_time > LOCK_DURATION:
            LOCK_ID = None

    return frame
