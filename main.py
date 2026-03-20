"""
main.py — Face Attendance System (Production)
Chạy: python main.py
      hoặc: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import cv2
import time
import asyncio
import threading
import sqlite3
import numpy as np
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from insightface.app import FaceAnalysis

# ── Thêm attendance_system vào path ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "attendance_system"))

from attendance_db      import init_db
from attendance_service import set_notify_queue, _push_notify
from telegram_notifier  import telegram_worker

# ════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════


RTSP_URL      = os.environ.get("RTSP_URL",
    "rtsp://admin:@192.168.1.69:554/user=admin&password=&channel=1&stream=0.sdp")
ROI_PATH      = os.environ.get("ROI_PATH",      "roi.npy")
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD",  "0.45"))
DET_THRESHOLD = float(os.environ.get("DET_THRESHOLD",  "0.6"))
MIN_FACE_PX   = int(os.environ.get("MIN_FACE_PX",      "60"))
DATASET_DIR   = os.environ.get("DATASET_DIR",   "dataset_test")

SQLITE_PATH   = os.environ.get("EMPLOYEES_DB",
    "/home/anhht/face_registration/data/employees.db")

os.environ.setdefault("ATTENDANCE_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/attendance.db"))
os.environ.setdefault("EMPLOYEES_DB", SQLITE_PATH)
os.environ.setdefault("PHOTOS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/photos"))
os.environ.setdefault("CHECKOUT_COOLDOWN", "300")
os.environ.setdefault("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
os.environ.setdefault("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID")


# ════════════════════════════════════════════
#  GLOBAL STATE
# ════════════════════════════════════════════
latest_frame    : object = None
frame_lock      = threading.Lock()
roi_coords      : object = None
db_names        : list   = []
db_embs         : np.ndarray  = np.zeros((0, 512), dtype=np.float32)
recording_state = {"active": False, "writer": None, "filename": None}


# ════════════════════════════════════════════
#  LOAD DB từ SQLite
# ════════════════════════════════════════════
def reload_db():
    global db_names, db_embs
    if not os.path.exists(SQLITE_PATH):
        print(f"[DB] employees.db chưa có tại {SQLITE_PATH}")
        db_names = []; db_embs = np.zeros((0,512), dtype=np.float32)
        return

    conn     = sqlite3.connect(SQLITE_PATH)
    rows     = conn.execute(
        "SELECT employee_id, embedding FROM face_embeddings ORDER BY employee_id"
    ).fetchall()
    emp_rows = conn.execute("SELECT employee_id, name FROM employees").fetchall()
    conn.close()

    id_to_name = {r[0]: r[1] for r in emp_rows}
    if not rows:
        db_names = []; db_embs = np.zeros((0,512), dtype=np.float32)
        print("[DB] Trống — chưa có nhân viên nào đăng ký"); return

    names, embs = [], []
    for employee_id, blob in rows:
        emb = np.frombuffer(blob, dtype=np.float32).copy()
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        names.append(id_to_name.get(employee_id, employee_id))
        embs.append(emb)

    db_names = names
    db_embs  = np.stack(embs).astype(np.float32)
    print(f"[DB] Loaded {len(db_names)} embeddings — {len(set(db_names))} người")
    for n in sorted(set(db_names)):
        print(f"  - {n}")


# ════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════
face_app = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection", "recognition"],
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("[MODEL] InsightFace buffalo_l loaded")


# ════════════════════════════════════════════
#  CAMERA THREAD
# ════════════════════════════════════════════
def camera_reader():
    global latest_frame
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[CAM] Reconnecting...")
            time.sleep(1); cap.open(RTSP_URL); continue
        if frame.ndim != 3 or frame.shape[0] < 10: continue
        with frame_lock:
            latest_frame = frame.copy()

cam_thread = threading.Thread(target=camera_reader, daemon=True)
cam_thread.start()
print("[CAM] Camera thread started")


# ════════════════════════════════════════════
#  ROI + RECOGNIZE
# ════════════════════════════════════════════
if os.path.exists(ROI_PATH):
    roi_coords = np.load(ROI_PATH).tolist()
    print(f"[ROI] Loaded: {roi_coords}")
else:
    print("[ROI] Chưa có ROI — vào /roi/select để chọn")


def recognize(face_emb: np.ndarray) -> tuple:
    if len(db_names) == 0: return "Unknown", 0.0
    sims  = db_embs @ face_emb
    idx   = int(np.argmax(sims))
    score = float(sims[idx])
    return (db_names[idx] if score >= SIM_THRESHOLD else "Unknown"), score


def scale_bbox_to_frame(x1, y1, x2, y2, roi_w, roi_h):
    rx1, ry1, rx2, ry2 = roi_coords
    sx = (rx2-rx1)/roi_w; sy = (ry2-ry1)/roi_h
    return (int(rx1+x1*sx), int(ry1+y1*sy), int(rx1+x2*sx), int(ry1+y2*sy))


# ════════════════════════════════════════════
#  BYTETRACKER
# ════════════════════════════════════════════
# ── Consecutive + Cooldown state ──
_consec_count  = {}   # {name: count}  — đếm frame liên tiếp
_last_seen     = {}   # {name: timestamp} — cooldown
CONSEC_K       = 5    # cần 5 frame liên tiếp
CHECKIN_COOLDOWN_SEC  = 86400  # 24h — check_in 1 lần/ngày
CHECKOUT_COOLDOWN_SEC = 300    # 5 phút — throttle check_out

print("[TRACKER] Consecutive+Cooldown mode ready")


# ════════════════════════════════════════════
#  APP LIFESPAN
# ════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    reload_db()
    init_db()
    notify_q = asyncio.Queue(maxsize=100)
    set_notify_queue(notify_q)
    tg_task  = asyncio.create_task(telegram_worker(notify_q))
    print("[APP] Startup complete")
    print("[APP] Endpoints: /video  /roi/select  /attendance/today  /health")
    yield
    tg_task.cancel()
    print("[APP] Shutdown")


app = FastAPI(title="Face Attendance System", lifespan=lifespan)


# ════════════════════════════════════════════
#  ENDPOINTS — ROI
# ════════════════════════════════════════════
@app.get("/snapshot")
def snapshot():
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        raise HTTPException(503, detail="Camera chưa sẵn sàng")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return StreamingResponse(iter([buf.tobytes()]), media_type="image/jpeg")


@app.get("/roi/select", response_class=HTMLResponse)
def roi_select_ui():
    return """<!DOCTYPE html><html><head><title>Chọn ROI</title>
<style>body{margin:0;background:#111;display:flex;flex-direction:column;
align-items:center;padding:20px;font-family:monospace;color:#eee}
canvas{cursor:crosshair;border:2px solid #0f0}
button{margin:8px;padding:8px 20px;background:#0a0;color:#fff;border:none;cursor:pointer}</style>
</head><body><h3>Kéo để chọn vùng ROI</h3>
<canvas id="c"></canvas><div id="info">Đang tải...</div>
<button onclick="saveROI()">Lưu ROI</button>
<button onclick="location.reload()">Tải lại</button>
<script>
const canvas=document.getElementById("c"),ctx=canvas.getContext("2d");
let img=new Image(),sx,sy,ex,ey,drawing=false,scaleX=1,scaleY=1;
img.onload=()=>{const maxW=Math.min(window.innerWidth-40,1200);
  scaleX=img.naturalWidth/Math.min(img.naturalWidth,maxW);
  scaleY=img.naturalHeight/(img.naturalHeight*maxW/img.naturalWidth);
  canvas.width=Math.min(img.naturalWidth,maxW);
  canvas.height=img.naturalHeight*canvas.width/img.naturalWidth;
  ctx.drawImage(img,0,0,canvas.width,canvas.height);
  document.getElementById("info").textContent=img.naturalWidth+"x"+img.naturalHeight;};
img.src="/snapshot?"+Date.now();
canvas.addEventListener("mousedown",e=>{const r=canvas.getBoundingClientRect();
  sx=e.clientX-r.left;sy=e.clientY-r.top;drawing=true;});
canvas.addEventListener("mousemove",e=>{if(!drawing)return;
  const r=canvas.getBoundingClientRect();ex=e.clientX-r.left;ey=e.clientY-r.top;
  ctx.drawImage(img,0,0,canvas.width,canvas.height);
  ctx.strokeStyle="#0f0";ctx.lineWidth=2;ctx.strokeRect(sx,sy,ex-sx,ey-sy);});
canvas.addEventListener("mouseup",()=>{drawing=false;});
function saveROI(){if(sx===undefined){alert("Chưa chọn!");return;}
  const x1=Math.round(Math.min(sx,ex)*scaleX),y1=Math.round(Math.min(sy,ey)*scaleY);
  const x2=Math.round(Math.max(sx,ex)*scaleX),y2=Math.round(Math.max(sy,ey)*scaleY);
  fetch("/roi/save",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({x1,y1,x2,y2})})
  .then(r=>r.json()).then(d=>{
    document.getElementById("info").textContent="Đã lưu ROI: ["+d.x1+","+d.y1+","+d.x2+","+d.y2+"]";});}
</script></body></html>"""


@app.post("/roi/save")
async def roi_save(request: Request):
    global roi_coords
    data = await request.json()
    roi_coords = [data["x1"], data["y1"], data["x2"], data["y2"]]
    np.save(ROI_PATH, np.array(roi_coords))
    return roi_coords


@app.get("/roi/current")
def roi_current():
    return {"roi": roi_coords}


# ════════════════════════════════════════════
#  ENDPOINTS — VIDEO STREAM
# ════════════════════════════════════════════

def _handle_attendance(name: str, score: float, frame, bbox):
    """Xử lý check-in/check-out sau khi consecutive đủ K frame."""
    import threading
    from attendance_db      import has_checkin_today, record_checkin, record_checkout, log_recognition_event
    from attendance_service import save_photo, _push_notify
    from datetime import datetime, date, timezone, timedelta

    VN_TZ = timezone(timedelta(hours=7))  # UTC+7
    import sqlite3 as _sq3

    now   = datetime.now(VN_TZ)
    today = now.date().isoformat()

    # Lấy employee_id từ employees.db
    employee_id = None
    try:
        _c = _sq3.connect(SQLITE_PATH)
        row = _c.execute("SELECT employee_id FROM employees WHERE name=? LIMIT 1",(name,)).fetchone()
        _c.close()
        if row: employee_id = row[0]
    except Exception: pass
    if employee_id is None:
        employee_id = name.lower().replace(" ","_")

    # Crop face để lưu ảnh
    x1,y1,x2,y2 = map(int, bbox)
    face_crop = frame[max(0,y1):y2, max(0,x1):x2]
    save_frame = face_crop if face_crop.size > 0 else frame

    already_in = has_checkin_today(employee_id, today)

    if not already_in:
        photo_path = save_photo(save_frame, employee_id, "checkin")
        record_checkin(employee_id, name, score, photo_path, now)
        log_recognition_event(employee_id, name, score, "check_in", photo_path)
        print(f"[CHECKIN]  {name:20s} score={score:.3f}")
        _push_notify({
            "type":"check_in","name":name,"employee_id":employee_id,
            "score":score,"time":now.strftime("%H:%M:%S"),
            "date":now.strftime("%d/%m/%Y"),"photo_path":photo_path,
        })
    else:
        last_co = _last_seen.get(f"co_{employee_id}")
        if last_co and (now - last_co).total_seconds() < CHECKOUT_COOLDOWN_SEC:
            return
        photo_path = save_photo(save_frame, employee_id, "checkout")
        record_checkout(employee_id, name, score, photo_path, now)
        _last_seen[f"co_{employee_id}"] = now
        log_recognition_event(employee_id, name, score, "check_out", photo_path)
        print(f"[CHECKOUT] {name:20s} score={score:.3f}")
        _push_notify({
            "type":"check_out","name":name,"employee_id":employee_id,
            "score":score,"time":now.strftime("%H:%M:%S"),
            "date":now.strftime("%d/%m/%Y"),"photo_path":photo_path,
        })


def gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.04); continue

        draw = frame.copy()
        h, w = frame.shape[:2]

        if roi_coords is not None:
            rx1=max(0,roi_coords[0]); ry1=max(0,roi_coords[1])
            rx2=min(w,roi_coords[2]); ry2=min(h,roi_coords[3])
            roi_frame = cv2.resize(frame[ry1:ry2, rx1:rx2], (640, 480))
            sx=(rx2-rx1)/640; sy=(ry2-ry1)/480; ox,oy=rx1,ry1
            # Vẽ khung ROI
            cv2.rectangle(draw,(rx1,ry1),(rx2,ry2),(255,165,0),2)
            cv2.putText(draw,"ROI",(rx1,ry1-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,165,0),1)
        else:
            roi_frame=cv2.resize(frame,(640,480)); sx=w/640; sy=h/480; ox,oy=0,0
            cv2.putText(draw,"ROI chua chon - vao /roi/select",(20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,165,255),2)

        faces = face_app.get(roi_frame)
        for face in faces:
            if face.det_score < DET_THRESHOLD: continue
            x1,y1,x2,y2 = map(int, face.bbox)
            if (x2-x1) < MIN_FACE_PX: continue

            name, score = recognize(face.normed_embedding)

            # Vẽ bbox
            fx1=int(ox+x1*sx); fy1=int(oy+y1*sy)
            fx2=int(ox+x2*sx); fy2=int(oy+y2*sy)
            color = (0,255,0) if name != "Unknown" else (0,0,255)
            cv2.rectangle(draw,(fx1,fy1),(fx2,fy2),color,2)
            cv2.putText(draw,f"{name} {score:.2f}",(fx1,fy1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            if name == "Unknown": continue

            # Consecutive counter
            _consec_count[name] = _consec_count.get(name, 0) + 1
            if _consec_count[name] < CONSEC_K:
                continue
            _consec_count[name] = 0

            # Lưu ảnh + ghi điểm danh
            _handle_attendance(name, score, roi_frame, face.bbox)

        if recording_state["active"]:
            cv2.circle(draw,(20,20),8,(0,0,255),-1)
            cv2.putText(draw,"REC",(32,26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

        _, buf = cv2.imencode(".jpg", draw, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
               + buf.tobytes() + b"\r\n")
        time.sleep(0.04)


@app.get("/video")
def video():
    return StreamingResponse(gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")


# ════════════════════════════════════════════
#  ENDPOINTS — ATTENDANCE
# ════════════════════════════════════════════
from attendance_db import get_attendance_today, get_attendance_by_date

@app.get("/attendance/today")
def attendance_today():
    return get_attendance_today()

@app.get("/attendance/{target_date}")
def attendance_by_date(target_date: str):
    return get_attendance_by_date(target_date)


# ════════════════════════════════════════════
#  ENDPOINTS — SYSTEM
# ════════════════════════════════════════════
@app.get("/health")
def health():
    with frame_lock:
        has_frame = latest_frame is not None
    return {
        "status"        : "ok",
        "camera"        : has_frame,
        "db_persons"    : len(set(db_names)),
        "db_embeddings" : len(db_names),
        "roi"           : roi_coords is not None,
        "mode": "consecutive_k5_cooldown",
    }

@app.post("/reload")
def reload():
    reload_db()
    return {"status": "ok", "persons": len(set(db_names))}


# ════════════════════════════════════════════
#  ENDPOINTS — RECORDING
# ════════════════════════════════════════════
@app.post("/record/start")
def record_start(video_name: str):
    if recording_state["active"]:
        raise HTTPException(400, detail="Đang ghi — gọi /record/stop trước")
    with frame_lock:
        frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        raise HTTPException(503, detail="Camera chưa sẵn sàng")
    filename = os.path.join(DATASET_DIR, video_name + ".mp4")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    h, w   = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    recording_state["writer"]   = cv2.VideoWriter(filename, fourcc, 20, (w,h))
    recording_state["filename"] = filename
    recording_state["active"]   = True
    def write_loop():
        while recording_state["active"]:
            with frame_lock:
                f = latest_frame.copy() if latest_frame is not None else None
            if f is not None: recording_state["writer"].write(f)
            time.sleep(1/20)
    threading.Thread(target=write_loop, daemon=True).start()
    return {"status": "recording", "file": filename}

@app.post("/record/stop")
def record_stop():
    if not recording_state["active"]:
        raise HTTPException(400, detail="Chưa bắt đầu ghi")
    recording_state["active"] = False
    time.sleep(0.2)
    recording_state["writer"].release()
    recording_state["writer"] = None
    return {"status": "saved", "file": recording_state["filename"]}

@app.get("/record/status")
def record_status():
    return {"active": recording_state["active"],
            "filename": recording_state["filename"]}


# ════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8501, reload=False)