const API_BASE = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws";

let ws = null;
let video = null;
let canvas = null;
let ctx = null;
let isSending = false;

const FRAME_INTERVAL = 120; 

document.addEventListener("DOMContentLoaded", () => {
    canvas = document.getElementById("videoCanvas");
    ctx = canvas.getContext("2d");

    initControls();
    initCamera();
    checkAPIStatus();
    setInterval(checkAPIStatus, 2000);
});

// ================= CAMERA =================
async function initCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 360 }
        });

        video = document.createElement("video");
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await video.play();

        canvas.width = 640;
        canvas.height = 360;

        connectWebSocket();
    } catch (err) {
        console.error("Camera error:", err);
        updateApiStatus("Camera Blocked", "error");
    }
}

// ================= WEBSOCKET =================
function connectWebSocket() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        console.log("[WS] Connected");
        isSending = false;
        sendFrameLoop();
    };

    ws.onmessage = event => {
        const img = new Image();
        img.onload = () => {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            isSending = false; 
        };
        img.src = "data:image/jpeg;base64," + event.data;
    };

    ws.onerror = err => {
        console.error("[WS ERROR]", err);
    };

    ws.onclose = () => {
        console.warn("[WS] Disconnected");
        isSending = false;
        setTimeout(connectWebSocket, 1000); 
    };
}

// ================= FRAME LOOP =================
function sendFrameLoop() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    if (!isSending) {
        isSending = true;

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame = canvas.toDataURL("image/jpeg", 0.6).split(",")[1];
        ws.send(frame);
    }

    setTimeout(sendFrameLoop, FRAME_INTERVAL);
}

// ================= CONTROLS =================
function initControls() {
    document.querySelectorAll(".mode-card").forEach(btn => {
        btn.addEventListener("click", async () => {
            const mode = btn.dataset.mode;
            await switchMode(mode);
        });
    });
}

async function switchMode(mode) {
    let endpoint = "/mode/stop";
    let label = "Standby";

    document.querySelectorAll(".mode-card").forEach(btn => btn.classList.remove("active"));

    if (mode === "1") {
        endpoint = "/mode/1";
        label = "Suspicious Activity Detection";
        document.getElementById("btn-sos").classList.add("active");
    } else if (mode === "2") {
        endpoint = "/mode/2";
        label = "Overcrowd Detection";
        document.getElementById("btn-crowd").classList.add("active");
    } else if (mode === "3") {
        endpoint = "/mode/3";
        label = "Target Lock System";
        document.getElementById("btn-target").classList.add("active");
    } else {
        document.getElementById("btn-stop").classList.add("active");
    }

    await fetch(API_BASE + endpoint, { method: "POST" });
    document.getElementById("current-mode").textContent = label;
}

// ================= STATUS =================
async function checkAPIStatus() {
    try {
        const res = await fetch(API_BASE + "/status");
        if (res.ok) updateApiStatus("Connected", "connected");
    } catch {
        updateApiStatus("Offline", "error");
    }
}

function updateApiStatus(text, status) {
    const el = document.getElementById("api-status");
    el.textContent = text;
    el.style.color =
        status === "connected" ? "#10B981" :
        status === "error" ? "#EF4444" :
        "#F59E0B";
}
