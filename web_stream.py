# import cv2
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# from insightface.app import FaceAnalysis

# RTSP_URL = "rtsp://xxxx/user=admin&password=&channel=1&stream=1.sdp"

# app = FastAPI()

# cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# face_app = FaceAnalysis(name="buffalo_l")
# face_app.prepare(ctx_id=0, det_size=(640, 640))

# def gen_frames():
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.resize(frame, (640, 480))
#         faces = face_app.get(frame)

#         for f in faces:
#             if f.det_score < 0.6:
#                 continue
#             x1, y1, x2, y2 = map(int, f.bbox)
#             cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

#         _, buffer = cv2.imencode(".jpg", frame)
#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n" +
#             buffer.tobytes() +
#             b"\r\n"
#         )

# @app.get("/video")
# def video_feed():
#     return StreamingResponse(gen_frames(),
#         media_type="multipart/x-mixed-replace; boundary=frame")

import os
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from insightface.app import FaceAnalysis

# ================== CONFIG ==================
RTSP_URL = "rtsp://admin:@192.168.1.69:554/user=admin&password=&channel=1&stream=0.sdp"
EMB_DIR = "data/embeddings"
SIM_THRESHOLD = 0.5
FRAME_W, FRAME_H = 640, 480
# ============================================

app = FastAPI()

# ---------- Load Face DB ----------
db_names = []
db_embs = []

for f in os.listdir(EMB_DIR):
    if f.endswith(".npy"):
        name = f.replace(".npy", "")
        emb = np.load(os.path.join(EMB_DIR, f))
    

        db_names.append(name)
        db_embs.append(emb)

if len(db_embs) == 0:
    raise RuntimeError("❌ Không có embedding nào trong data/embeddings")

db_embs = np.stack(db_embs)
print("✅ Loaded face DB:", db_names)

# ---------- Face Model ----------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ---------- Video ----------
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

def recognize(face_emb):
    sims = db_embs @ face_emb
    idx = np.argmax(sims)
    score = sims[idx]

    if score >= SIM_THRESHOLD:
        return db_names[idx], float(score)
    return "UNKNOWN", float(score)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        faces = face_app.get(frame)

        for f in faces:
            if f.det_score < 0.6:
                continue

            emb = f.normed_embedding  # (512,) đã norm
            name, score = recognize(emb)

            x1, y1, x2, y2 = map(int, f.bbox)
            color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{name} {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

@app.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
