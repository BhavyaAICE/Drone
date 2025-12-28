import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import deque
from twilio.rest import Client
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import time

# ==================== TWILIO CONFIG ====================
ACCOUNT_SID = "ACa9c3b27777fb0bb6cce3c3b3dc1adff5"
AUTH_TOKEN  = "04d0af6341854f1177360b7ca6078b51"
TWILIO_NUMBER = "+14843020213"
YOUR_NUMBER   = "+918209043296"
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# ==================== SUSPICIOUS HAND/SPRINT CONFIG ====================
RUN_THRESHOLD = 25
HISTORY_LEN = 6
last_sos_time = 0

# ==================== OVERCROWD CONFIG ====================
OVERCROWD_LIMIT = 8
SMS_COOLDOWN = 60
DOT_RADIUS = 5
DOT_COLOR = (0, 255, 0)
ALERT_COLOR = (0, 0, 255)
last_sms_time = 0

# ==================== TARGET LOCK CONFIG ====================
LOCK_ID = None
lock_time = 0
LOCK_DURATION = 4
os.makedirs("screenshots", exist_ok=True)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

target_img = cv2.imread("SRK.jpg", cv2.IMREAD_GRAYSCALE)
if target_img is None:
    raise Exception("Target image not found")
faces = face_cascade.detectMultiScale(target_img, 1.3, 5)
if len(faces) == 0:
    raise Exception("No face in target image")
(x, y, w, h) = faces[0]
target_face = cv2.resize(target_img[y:y+h, x:x+w], (200, 200))
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train([target_face], np.array([1]))

# ==================== YOLO & TRACKER ====================
yolo = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
track_history = {}

# ==================== HELPER FUNCTIONS ====================
def send_dummy_sos():
    global last_sos_time
    if time.time() - last_sos_time < 30:
        return
    last_sos_time = time.time()
    message = (
        "ALERT FROM DETECTOR\n"
        "Suspicious activity detected recently \n\n"
        "📍 Location: CDTI Jaipur , Dahmi kalan, Manipal University Rd, Bagru, Jaipur, Rajasthan\n"
        "🧭 Coordinates: 26.846472,75.558438\n")
    client.messages.create(body=message, from_=TWILIO_NUMBER, to=YOUR_NUMBER)
    print("[SOS SENT] Fatal behaviour detected")

def send_overcrowd_sms(count):
    global last_sms_time
    if time.time() - last_sms_time < SMS_COOLDOWN:
        return
    last_sms_time = time.time()
    msg = (
        f"🚨 OVERCROWD ALERT 🚨\n"
        f"People detected: {count}\n"
        "📍 Location: CDTI Jaipur, Dahmi Kalan, Manipal University Rd, Bagru, Jaipur, Rajasthan\n"
        "🧭 Coordinates: 26.846472,75.558438"
    )
    try:
        client.messages.create(body=msg, from_=TWILIO_NUMBER, to=YOUR_NUMBER)
        print("[SMS SENT] Overcrowd alert")
    except Exception as e:
        print("SMS failed:", e)

# ==================== MAIN LOOP ====================
cap = cv2.VideoCapture(0)
mode = None  # Modes: "sos", "crowd", "target"

print("Press '1' for Suspicious Detection + SOS")
print("Press '2' for Overcrowd Detection")
print("Press '3' for Target Lock System")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    key = cv2.waitKey(1) & 0xFF

    # SWITCH MODES
    if key == ord("1"):
        mode = "sos"
        print("[MODE] Suspicious Detection + SOS Active")
    elif key == ord("2"):
        mode = "crowd"
        print("[MODE] Overcrowd Detection Active")
    elif key == ord("3"):
        mode = "target"
        print("[MODE] Target Lock System Active")
    elif key == ord("q"):
        break

    if mode == "sos":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo.predict(frame, conf=0.5, classes=[0], verbose=False)
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                suspicious = False
                label = "No Harm"
                color = (0, 255, 0)
                crop = rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                pose_result = pose.process(crop)
                if pose_result.pose_landmarks:
                    lm = pose_result.pose_landmarks.landmark
                    lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                    rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                    ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    if lw.y < ls.y or rw.y < rs.y:
                        suspicious = True
                        label = "Fatal Gesture"
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2
                pid = (x1, y1, x2, y2)
                if pid not in track_history:
                    track_history[pid] = deque(maxlen=HISTORY_LEN)
                track_history[pid].append((cx, cy))
                if len(track_history[pid]) >= 2:
                    dx = track_history[pid][-1][0] - track_history[pid][0][0]
                    dy = track_history[pid][-1][1] - track_history[pid][0][1]
                    speed = np.sqrt(dx*dx + dy*dy)
                    if speed > RUN_THRESHOLD:
                        suspicious = True
                        label = "Running or Panic"
                if suspicious:
                    color = (0,0,255)
                    send_dummy_sos()
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
                cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    elif mode == "crowd":
        results = yolo.predict(frame, imgsz=640, conf=0.5, classes=[0])
        total_people = 0
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                total_people += 1
                cx = (x1 + x2)//2
                cy = (y1 + y2)//2
                cv2.circle(frame, (cx, cy), DOT_RADIUS, DOT_COLOR, -1)
        if total_people >= OVERCROWD_LIMIT:
            cv2.putText(frame, "OVERCROWD DETECTED!", (30,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, ALERT_COLOR, 3)
            send_overcrowd_sms(total_people)
        cv2.putText(frame, f"People Count: {total_people}", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    ALERT_COLOR if total_people>=OVERCROWD_LIMIT else (0,255,0), 2)

    elif mode == "target":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = yolo(frame, conf=0.5, verbose=False)[0]
        detections = []
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2-x1, y2-y1], 0.9, "person"))
        tracks = tracker.update_tracks(detections, frame=frame)
        active_ids = set()
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            active_ids.add(tid)
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            person = gray[y1:y2, x1:x2]
            faces = face_cascade.detectMultiScale(person, 1.3, 5)
            for (fx, fy, fw, fh) in faces:
                if fw < 80 or fh < 80:
                    continue
                face_crop = cv2.resize(person[fy:fy+fh, fx:fx+fw], (200,200))
                _, confidence = recognizer.predict(face_crop)
                if confidence < 80:
                    LOCK_ID = tid
                    lock_time = time.time()
                    cv2.imwrite(f"screenshots/TARGET_{int(time.time())}.jpg", frame)
            if LOCK_ID is not None and tid == LOCK_ID:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255),3)
                cv2.putText(frame,"TARGET LOCKED",(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        if LOCK_ID is not None and LOCK_ID not in active_ids:
            if time.time() - lock_time > LOCK_DURATION:
                LOCK_ID = None

    cv2.imshow("Integrated Security System", frame)

cap.release()
cv2.destroyAllWindows()