import os
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

# ================== CONFIG ==================
RTSP_URL      = "rtsp://admin:@192.168.1.69:554/user=admin&password=&channel=1&stream=0.sdp"
EMB_DIR       = "data/embeddings"
ROI_PATH      = "roi.npy"
SIM_THRESHOLD = 0.5
ROI_TARGET_W  = 640
ROI_TARGET_H  = 480
# ============================================

app = FastAPI()

# ---------- Load Face DB ----------
db_names, db_embs = [], []
for f in os.listdir(EMB_DIR):
    if f.endswith(".npy"):
        name = f.replace(".npy", "")
        emb  = np.load(os.path.join(EMB_DIR, f))
        db_names.append(name)
        db_embs.append(emb)

if len(db_embs) == 0:
    raise RuntimeError("Không có embedding nào trong data/embeddings")

db_embs = np.stack(db_embs)
print("Loaded face DB:", db_names)

# ---------- Load ROI (nếu có) ----------
roi_coords = None
if os.path.exists(ROI_PATH):
    roi_coords = np.load(ROI_PATH).astype(int)
    rx1, ry1, rx2, ry2 = roi_coords
    print(f"ROI loaded: x1={rx1} y1={ry1} x2={rx2} y2={ry2}")
else:
    print("Chưa có ROI — truy cập /roi/select để chọn")

# ---------- Face Model ----------
from insightface.app import FaceAnalysis
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ---------- Camera ----------
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# ---------- Helpers ----------
def recognize(face_emb):
    sims  = db_embs @ face_emb
    idx   = np.argmax(sims)
    score = sims[idx]
    if score >= SIM_THRESHOLD:
        return db_names[idx], float(score)
    return "UNKNOWN", float(score)


def scale_bbox_to_full(x1, y1, x2, y2, roi_w, roi_h):
    rx1, ry1, rx2, ry2 = roi_coords
    scale_x = (rx2 - rx1) / roi_w
    scale_y = (ry2 - ry1) / roi_h
    return (
        int(rx1 + x1 * scale_x),
        int(ry1 + y1 * scale_y),
        int(rx1 + x2 * scale_x),
        int(ry1 + y2 * scale_y),
    )


def read_frame():
    ret, frame = cap.read()
    if not ret:
        cap.open(RTSP_URL)
        ret, frame = cap.read()
    return ret, frame


# ==========================================
# ROI SELECTOR ENDPOINTS
# ==========================================

@app.get("/roi/select", response_class=HTMLResponse)
def roi_select_ui():
    """Giao diện browser để kéo chuột chọn ROI."""
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>ROI Selector</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Courier New', monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 24px;
    gap: 16px;
    min-height: 100vh;
  }
  h1 {
    font-size: 13px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00ff99;
    border-bottom: 1px solid #00ff9933;
    padding-bottom: 8px;
    width: 100%;
    text-align: center;
  }
  #hint {
    font-size: 11px;
    color: #666;
    letter-spacing: 0.1em;
  }
  #canvas-wrap {
    position: relative;
    border: 1px solid #1a1a2e;
    box-shadow: 0 0 40px #00ff9911;
  }
  canvas { display: block; cursor: crosshair; }
  #coords {
    font-size: 12px;
    color: #00ff99;
    letter-spacing: 0.1em;
    height: 18px;
  }
  #btn-save {
    background: transparent;
    border: 1px solid #00ff99;
    color: #00ff99;
    padding: 8px 32px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    cursor: pointer;
    transition: background 0.2s;
    display: none;
  }
  #btn-save:hover { background: #00ff9922; }
  #status {
    font-size: 11px;
    letter-spacing: 0.1em;
    height: 16px;
  }
  .ok  { color: #00ff99; }
  .err { color: #ff4444; }
</style>
</head>
<body>
<h1>ROI Selector</h1>
<span id="hint">Kéo chuột để chọn vùng cửa → nhấn SAVE</span>

<div id="canvas-wrap">
  <canvas id="c"></canvas>
</div>

<span id="coords">—</span>
<button id="btn-save" onclick="saveROI()">SAVE ROI</button>
<span id="status"></span>

<script>
const canvas  = document.getElementById('c');
const ctx     = canvas.getContext('2d');
const coordEl = document.getElementById('coords');
const statusEl= document.getElementById('status');
const btnSave = document.getElementById('btn-save');

const img = new Image();
img.src = '/snapshot';
img.onload = () => {
  canvas.width  = img.naturalWidth;
  canvas.height = img.naturalHeight;
  ctx.drawImage(img, 0, 0);
};

let startX, startY, endX, endY, dragging = false;

function getPos(e) {
  const r = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / r.width;
  const scaleY = canvas.height / r.height;
  return {
    x: Math.round((e.clientX - r.left) * scaleX),
    y: Math.round((e.clientY - r.top)  * scaleY),
  };
}

canvas.addEventListener('mousedown', e => {
  const p = getPos(e);
  startX = p.x; startY = p.y;
  dragging = true;
  btnSave.style.display = 'none';
  statusEl.textContent = '';
});

canvas.addEventListener('mousemove', e => {
  if (!dragging) return;
  const p = getPos(e);
  endX = p.x; endY = p.y;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  const x = Math.min(startX, endX), y = Math.min(startY, endY);
  const w = Math.abs(endX - startX),  h = Math.abs(endY - startY);
  ctx.strokeStyle = '#00ff99';
  ctx.lineWidth   = 2;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = 'rgba(0,255,153,0.08)';
  ctx.fillRect(x, y, w, h);
  coordEl.textContent = `x1=${x}  y1=${y}  x2=${x+w}  y2=${y+h}  (${w}×${h}px)`;
});

canvas.addEventListener('mouseup', e => {
  if (!dragging) return;
  dragging = false;
  const p = getPos(e);
  endX = p.x; endY = p.y;
  if (Math.abs(endX-startX) > 10 && Math.abs(endY-startY) > 10) {
    btnSave.style.display = 'inline-block';
  }
});

async function saveROI() {
  const x1 = Math.min(startX, endX), y1 = Math.min(startY, endY);
  const x2 = Math.max(startX, endX), y2 = Math.max(startY, endY);
  const res = await fetch('/roi/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x1, y1, x2, y2 }),
  });
  const data = await res.json();
  if (data.ok) {
    statusEl.className = 'ok';
    statusEl.textContent = `ROI saved: x1=${x1} y1=${y1} x2=${x2} y2=${y2}`;
  } else {
    statusEl.className = 'err';
    statusEl.textContent = 'Lỗi: ' + data.error;
  }
}
</script>
</body>
</html>
""")


@app.get("/snapshot")
def snapshot():
    """Trả về 1 frame JPEG từ camera để hiển thị trên UI chọn ROI."""
    ret, frame = read_frame()
    if not ret:
        return JSONResponse({"error": "Không đọc được frame"}, status_code=500)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")


@app.post("/roi/save")
async def roi_save(payload: dict):
    """Nhận tọa độ từ browser, lưu roi.npy, cập nhật roi_coords."""
    global roi_coords
    try:
        x1 = int(payload["x1"]); y1 = int(payload["y1"])
        x2 = int(payload["x2"]); y2 = int(payload["y2"])
        if x2 <= x1 or y2 <= y1:
            return JSONResponse({"ok": False, "error": "Tọa độ không hợp lệ"})
        coords = np.array([x1, y1, x2, y2])
        np.save(ROI_PATH, coords)
        roi_coords = coords
        print(f"ROI updated: x1={x1} y1={y1} x2={x2} y2={y2}")
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.get("/roi/current")
def roi_current():
    """Trả về ROI hiện tại."""
    if roi_coords is None:
        return JSONResponse({"ok": False, "error": "Chưa có ROI"})
    x1, y1, x2, y2 = roi_coords
    return JSONResponse({"ok": True, "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})


# ==========================================
# VIDEO STREAM
# ==========================================

def gen_frames():
    while True:
        ret, frame = read_frame()
        if not ret:
            continue

        if roi_coords is not None:
            rx1, ry1, rx2, ry2 = roi_coords

            roi_crop    = frame[ry1:ry2, rx1:rx2]
            roi_resized = cv2.resize(
                roi_crop,
                (ROI_TARGET_W, ROI_TARGET_H),
                interpolation=cv2.INTER_LINEAR,
            )
            # print(f"ROI crop: {roi_crop.shape[1]}×{roi_crop.shape[0]}")
            # print(f"ROI resized: {roi_resized.shape[1]}×{roi_resized.shape[0]}")
            
            # faces = face_app.get(roi_resized)
            # for f in faces:
            #     if f.det_score < 0.6:
            #         continue
            #     face_w = int(f.bbox[2]) - int(f.bbox[0])
                
            #     # THÊM VÀO ĐÂY
            #     print(f"face width in resized: {face_w}px | det_score: {f.det_score:.2f}")

            

            faces = face_app.get(roi_resized)
            for f in faces:
                if f.det_score < 0.6:
                    continue
                face_w = int(f.bbox[2]) - int(f.bbox[0])
                if face_w < 60:
                    continue

                name, score = recognize(f.normed_embedding)
                x1, y1, x2, y2 = scale_bbox_to_full(
                    *map(int, f.bbox), ROI_TARGET_W, ROI_TARGET_H
                )
                color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{name} {score:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

            # Vẽ ROI boundary
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        else:
            # Chưa có ROI → chạy toàn frame, hiện warning
            cv2.putText(
                frame, "ROI chua duoc chon — truy cap /roi/select",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
            )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )