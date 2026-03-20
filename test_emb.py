import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# Đọc ảnh test (ảnh mặt mày)
img = cv2.imread("/home/anhht/face-attendance-mvp/data/employees/tuananh.png")  # đổi đúng path ảnh

faces = app.get(img)

if len(faces) == 0:
    print("❌ Không phát hiện khuôn mặt")
    exit()

f = faces[0]

emb = f.embedding
normed_emb = f.normed_embedding

print("===== CHECK EMBEDDING =====")
print("Embedding shape:", emb.shape)
print("Embedding norm (RAW):", np.linalg.norm(emb))
print("Normed embedding norm:", np.linalg.norm(normed_emb))

# Manual normalize để so
emb_manual = emb / np.linalg.norm(emb)
print("Manual norm:", np.linalg.norm(emb_manual))

# Cosine check
cos1 = np.dot(normed_emb, emb_manual)
cos2 = np.dot(emb_manual, emb_manual)

print("Cosine(normed, manual):", cos1)
print("Cosine(manual, manual):", cos2)
