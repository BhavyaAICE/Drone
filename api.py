import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import drone

app = FastAPI()

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- RESPONSE MODEL ----------
class StatusResponse(BaseModel):
    current_mode: str | None
    status: str

# ---------- STATUS ----------
@app.get("/status")
def get_status():
    with drone.mode_lock:
        current = drone.mode
    return StatusResponse(current_mode=current, status="running")

# ---------- MODE ENDPOINTS ----------
@app.post("/mode/1")
def set_suspicious_detection():
    with drone.mode_lock:
        drone.mode = "sos"
    return {"status": "Suspicious Detection + SOS Active", "mode": "sos"}

@app.post("/mode/2")
def set_overcrowd_detection():
    with drone.mode_lock:
        drone.mode = "crowd"
    return {"status": "Overcrowd Detection Active", "mode": "crowd"}

@app.post("/mode/3")
def set_target_lock():
    with drone.mode_lock:
        drone.mode = "target"
    return {"status": "Target Lock System Active", "mode": "target"}

@app.post("/mode/stop")
def stop_mode():
    with drone.mode_lock:
        drone.mode = None
    return {"status": "Detection Stopped", "mode": None}

# ---------- WEBSOCKET ----------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("[WS] Client connected")

    try:
        while True:
            data = await ws.receive_text()

            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            with drone.mode_lock:
                active_mode = drone.mode

            processed = drone.process_frame(frame)

            _, buffer = cv2.imencode(".jpg", processed)
            encoded = base64.b64encode(buffer).decode("utf-8")

            await ws.send_text(encoded)


            await asyncio.sleep(0.12)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")



# ---------- START SERVER ----------
def start():
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )

if __name__ == "__main__":
    start()
