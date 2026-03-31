"""
ThreatSight - Demo API
Lightweight demo version that runs without ML dependencies (YOLOv5, ONNX, DeepFace, PyTorch).
All detections and recognitions are simulated with realistic outputs.
"""

import base64
import hashlib
import random
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ThreatSight - Demo",
    description="AI-powered threat detection system (demo mode — simulated detections)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


@app.middleware("http")
async def add_demo_header(request: Request, call_next):
    response: Response = await call_next(request)
    response.headers["X-Demo-Mode"] = "true"
    return response


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    image_url: str | None = Field(None, description="URL of image to analyze")
    image_base64: str | None = Field(None, description="Base64-encoded image data")
    camera_id: str = Field("cam_lobby", description="Source camera ID")
    confidence_threshold: float = Field(0.45, ge=0.1, le=1.0)


class RecognizeRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    camera_id: str = "cam_lobby"


class AuthRequest(BaseModel):
    username: str
    password: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAFE_OBJECTS = [
    ("person", 0.92), ("person", 0.87), ("car", 0.94), ("backpack", 0.78),
    ("handbag", 0.71), ("bicycle", 0.88), ("dog", 0.83), ("cell_phone", 0.69),
    ("umbrella", 0.74), ("suitcase", 0.80),
]

THREAT_OBJECTS = [
    ("knife", 0.72), ("firearm", 0.68), ("rifle", 0.61),
]

KNOWN_FACES = [
    {"face_id": "face_001", "name": "John Doe", "role": "employee", "department": "Engineering", "clearance": "L3", "registered": "2024-06-15T09:00:00Z"},
    {"face_id": "face_002", "name": "Jane Smith", "role": "employee", "department": "Security", "clearance": "L5", "registered": "2024-03-20T14:30:00Z"},
    {"face_id": "face_003", "name": "Robert Chen", "role": "visitor", "department": None, "clearance": "L1", "registered": "2025-01-10T11:15:00Z"},
    {"face_id": "face_004", "name": "Maria Garcia", "role": "contractor", "department": "Facilities", "clearance": "L2", "registered": "2024-09-01T08:00:00Z"},
    {"face_id": "face_005", "name": "Alex Kim", "role": "employee", "department": "Research", "clearance": "L4", "registered": "2024-11-22T16:45:00Z"},
]

CAMERAS = [
    {
        "camera_id": "cam_lobby",
        "name": "Main Lobby",
        "location": "Building A — Ground Floor Entrance",
        "resolution": "1920x1080",
        "fps": 30,
        "status": "online",
        "last_frame": None,  # filled dynamically
    },
    {
        "camera_id": "cam_entrance",
        "name": "Parking Entrance",
        "location": "Building A — East Gate",
        "resolution": "2560x1440",
        "fps": 25,
        "status": "online",
        "last_frame": None,
    },
]

ALERT_TEMPLATES = [
    {"severity": "CRITICAL", "description": "Potential firearm detected in lobby camera feed"},
    {"severity": "HIGH", "description": "Unidentified individual in restricted zone B"},
    {"severity": "MEDIUM", "description": "Unusual loitering detected near east entrance"},
    {"severity": "LOW", "description": "Unregistered face detected at reception"},
    {"severity": "INFO", "description": "Camera cam_entrance reconnected after brief dropout"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_bbox() -> dict:
    x = random.randint(50, 1600)
    y = random.randint(50, 900)
    w = random.randint(40, 250)
    h = random.randint(60, 350)
    return {"x": x, "y": y, "width": w, "height": h}


def _generate_detections(seed: str | None = None, threshold: float = 0.45) -> dict:
    if seed:
        random.seed(hash(seed) % 2**32)
    else:
        random.seed(None)

    num = random.randint(2, 6)
    detections = []
    has_threat = random.random() < 0.10  # 10 % chance

    for i in range(num):
        label, conf = random.choice(SAFE_OBJECTS)
        if conf < threshold:
            continue
        detections.append({
            "detection_id": str(uuid.uuid4()),
            "label": label,
            "confidence": round(conf + random.uniform(-0.05, 0.05), 3),
            "bbox": _rand_bbox(),
            "is_threat": False,
        })

    if has_threat:
        t_label, t_conf = random.choice(THREAT_OBJECTS)
        detections.append({
            "detection_id": str(uuid.uuid4()),
            "label": t_label,
            "confidence": round(t_conf + random.uniform(-0.03, 0.08), 3),
            "bbox": _rand_bbox(),
            "is_threat": True,
        })

    random.seed(None)
    threat_detected = any(d["is_threat"] for d in detections)
    return {
        "frame_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detections": detections,
        "detection_count": len(detections),
        "threat_detected": threat_detected,
        "alert": {
            "alert_id": str(uuid.uuid4()),
            "severity": "CRITICAL",
            "message": f"Threat object detected: {next((d['label'] for d in detections if d['is_threat']), 'unknown')}",
        } if threat_detected else None,
        "processing_ms": round(random.uniform(18, 55), 1),
        "model": "YOLOv5x-threat-v3.2 (demo)",
    }


def _generate_alerts(n: int = 5) -> list[dict]:
    now = datetime.now(timezone.utc)
    alerts = []
    for i in range(n):
        tpl = ALERT_TEMPLATES[i % len(ALERT_TEMPLATES)]
        ts = now - timedelta(minutes=random.randint(3, 480))
        alerts.append({
            "alert_id": str(uuid.uuid4()),
            "timestamp": ts.isoformat(),
            "severity": tpl["severity"],
            "camera_id": random.choice(["cam_lobby", "cam_entrance"]),
            "description": tpl["description"],
            "acknowledged": random.choice([True, False]),
        })
    return sorted(alerts, key=lambda a: a["timestamp"], reverse=True)


def _make_token(username: str) -> str:
    payload = f'{{"sub":"{username}","mode":"demo","iat":{int(time.time())}}}'
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "name": "ThreatSight",
        "version": "1.0.0",
        "status": "running",
        "mode": "demo",
        "docs_url": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "mode": "demo",
    }


@app.post("/api/v1/detect")
async def detect(req: DetectRequest):
    seed = req.image_url or req.image_base64 or str(uuid.uuid4())
    return _generate_detections(seed=seed, threshold=req.confidence_threshold)


@app.get("/api/v1/faces")
async def list_faces():
    return {"faces": KNOWN_FACES, "total": len(KNOWN_FACES)}


@app.post("/api/v1/recognize")
async def recognize(req: RecognizeRequest):
    matched = random.random() < 0.6
    face = random.choice(KNOWN_FACES) if matched else None
    return {
        "frame_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "faces_found": random.randint(1, 3),
        "match": {
            "face_id": face["face_id"],
            "name": face["name"],
            "role": face["role"],
            "confidence": round(random.uniform(0.82, 0.98), 3),
        } if face else None,
        "matched": matched,
        "processing_ms": round(random.uniform(40, 120), 1),
        "model": "DeepFace-ArcFace (demo)",
    }


@app.get("/api/v1/alerts")
async def get_alerts():
    return {"alerts": _generate_alerts(5), "total": 5}


@app.get("/api/v1/cameras")
async def list_cameras():
    now = datetime.now(timezone.utc).isoformat()
    cams = []
    for c in CAMERAS:
        cam = dict(c)
        cam["last_frame"] = now
        cams.append(cam)
    return {"cameras": cams, "total": len(cams)}


@app.get("/api/v1/analytics/dashboard")
async def dashboard():
    return {
        "period": "last_24h",
        "detections_total": random.randint(1200, 3500),
        "threats_detected": random.randint(0, 4),
        "faces_recognized": random.randint(80, 250),
        "unknown_faces": random.randint(10, 40),
        "active_cameras": 2,
        "cameras_offline": 0,
        "alerts_by_severity": {
            "CRITICAL": random.randint(0, 2),
            "HIGH": random.randint(1, 5),
            "MEDIUM": random.randint(3, 12),
            "LOW": random.randint(5, 20),
            "INFO": random.randint(10, 30),
        },
        "avg_processing_ms": round(random.uniform(22, 45), 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/auth/token")
async def auth_token(req: AuthRequest):
    return {
        "access_token": _make_token(req.username),
        "token_type": "bearer",
        "expires_in": 3600,
        "mode": "demo",
        "note": "Demo mode — any credentials are accepted",
    }
