import cv2
from insightface.app import FaceAnalysis

img_path = "test_frame.jpg"   # ảnh bạn đã chụp từ RTSP

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

img = cv2.imread(img_path)
faces = app.get(img)

print(f"Detected faces: {len(faces)}")

for i, face in enumerate(faces):
    print(f"\nFace #{i+1}")
    print("  bbox :", face.bbox)
    print("  score:", face.det_score)
    print("  emb norm:", (face.embedding**2).sum() ** 0.5)
