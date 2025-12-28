from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
import drone

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModeRequest(BaseModel):
    mode: str

class StatusResponse(BaseModel):
    current_mode: str | None
    status: str

@app.get("/status")
def get_status():
    return StatusResponse(
        current_mode=drone.mode,
        status="running"
    )

@app.post("/mode/1")
def set_suspicious_detection():
    drone.mode = "sos"
    print("[API] Switched to Suspicious Detection + SOS")
    return {"status": "Suspicious Detection + SOS Active", "mode": "sos"}

@app.post("/mode/2")
def set_overcrowd_detection():
    drone.mode = "crowd"
    print("[API] Switched to Overcrowd Detection")
    return {"status": "Overcrowd Detection Active", "mode": "crowd"}

@app.post("/mode/3")
def set_target_lock():
    drone.mode = "target"
    print("[API] Switched to Target Lock System")
    return {"status": "Target Lock System Active", "mode": "target"}

@app.post("/mode/stop")
def stop_mode():
    drone.mode = None
    print("[API] Stopped all detection")
    return {"status": "Detection Stopped", "mode": None}

def run_api():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    drone.run_detection_loop()
