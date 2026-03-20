# import cv2
# import time
# from insightface.app import FaceAnalysis

# RTSP_URL = "rtsp://admin:@192.168.1.69:554/user=admin&password=&channel=1&stream=1.sdp"

# face_app = FaceAnalysis(
#     name="buffalo_l",
#     providers=["CUDAExecutionProvider"]
# )
# face_app.prepare(ctx_id=0, det_size=(640, 640))

# cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

# # while True:
# #     t0 = time.time()
# #     ret, frame = cap.read()
# #     t1 = time.time()

# #     if not ret:
# #         break

# #     faces = face_app.get(frame)
# #     t2 = time.time()

# #     print(
# #         f"read: {(t1-t0)*1000:.1f} ms | "
# #         f"infer: {(t2-t1)*1000:.1f} ms | "
# #         f"faces: {len(faces)}"
# #     )


# frame_count = 0
# start_time = time.time()

# while True:
#     t0 = time.time()
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_count += 1

#     # đo mỗi 1 giây
#     if time.time() - start_time >= 1.0:
#         print("Real FPS:", frame_count)
#         frame_count = 0
#         start_time = time.time()

#     faces = face_app.get(frame)

import cv2
import time
from insightface.app import FaceAnalysis

RTSP_URL = "rtsp://admin:@192.168.1.69:554/user=admin&password=&channel=1&stream=0.sdp"

# ===============================
# INIT MODEL
# ===============================
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider"]
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ===============================
# OPEN CAMERA
# ===============================
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ Cannot open RTSP stream")
    exit()

print("🚀 Measuring FPS... (Ctrl+C to stop)\n")

# ===============================
# FPS VARIABLES
# ===============================
cam_frames = 0
pipeline_frames = 0
infer_time_total = 0.0

measure_interval = 5.0  # đo mỗi 5 giây cho ổn định
start_time = time.time()

try:
    while True:
        loop_start = time.time()

        # -------- READ FRAME --------
        t0 = time.time()
        ret, frame = cap.read()
        t1 = time.time()

        if not ret:
            print("⚠ Stream ended")
            break

        cam_frames += 1

        # -------- INFERENCE --------
        faces = face_app.get(frame)
        t2 = time.time()

        infer_time_total += (t2 - t1)
        pipeline_frames += 1

        # -------- REPORT --------
        elapsed = time.time() - start_time
        if elapsed >= measure_interval:

            cam_fps = cam_frames / elapsed
            pipeline_fps = pipeline_frames / elapsed
            infer_fps = (
                pipeline_frames / infer_time_total
                if infer_time_total > 0 else 0
            )

            print("======================================")
            print(f"⏱  Duration          : {elapsed:.2f} s")
            print(f"🎥 Camera FPS        : {cam_fps:.2f}")
            print(f"🧠 Inference FPS     : {infer_fps:.2f}")
            print(f"⚙ Full Pipeline FPS : {pipeline_fps:.2f}")
            print(f"👤 Last faces count  : {len(faces)}")
            print("======================================\n")

            # reset
            cam_frames = 0
            pipeline_frames = 0
            infer_time_total = 0.0
            start_time = time.time()

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")

cap.release()