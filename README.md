# Hệ thống điểm danh nhận diện khuôn mặt

> Nhận diện khuôn mặt realtime từ camera RTSP tại cửa ra vào, tự động ghi nhận giờ vào/ra và gửi thông báo lên nhóm Telegram công ty.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![InsightFace](https://img.shields.io/badge/InsightFace-buffalo__l-orange)
![SQLite](https://img.shields.io/badge/Database-SQLite-lightgrey)

---

## Demo

| Đăng ký khuôn mặt | Nhận diện realtime | Thông báo Telegram |
|---|---|---|
| ![register](docs/register.png) | ![detection](docs/detection.png) | ![telegram](docs/telegram.png) |

---

## Bài toán

Điểm danh thủ công tốn thời gian và hay xảy ra sai sót. Hệ thống này tự động hóa quy trình bằng camera cố định tại cửa ra vào — nhân viên đi qua sẽ được tự động ghi nhận giờ vào/ra, kèm ảnh và thông báo tức thì lên nhóm Telegram của công ty, không cần thao tác gì thêm.

---

## Kiến trúc hệ thống

```
┌─────────────────────────┐           ┌──────────────────────────────┐
│   App đăng ký           │           │   App điểm danh              │
│   face_registration     │──SQLite──▶│   face-attendance-mvp        │
│   port 8001             │           │   port 8501                  │
│   Nhân viên tự chụp ảnh │           │   Nhận diện từ camera RTSP   │
└─────────────────────────┘           └──────────────────────────────┘
          │                                         │
          ▼                                         ▼
   employees.db                             attendance.db
   face_embeddings                           attendance_log
   (512 chiều, ArcFace)                      recognition_events
                                                    │
                                                    ▼
                                            Nhóm Telegram
```

### Luồng dữ liệu điểm danh

```
Camera RTSP 25FPS
      │
      ▼  [Thread riêng]
  latest_frame (luôn là frame mới nhất)
      │
      ▼  [Mỗi 40ms]
  Crop vùng ROI → Resize 640x480
      │
      ▼  InsightFace buffalo_l
  RetinaFace  → phát hiện khuôn mặt + 5 điểm mốc
  ArcFace R50 → vector đặc trưng 512 chiều
      │
      ▼  Cosine similarity với DB
  score >= 0.45 → tên nhân viên
      │
      ▼  Xác nhận liên tiếp (5 frame liên tiếp)
  _handle_attendance()
      │
      ├── Lần đầu trong ngày → CHECK-IN  → ghi DB
      └── Đã check-in rồi   → CHECK-OUT → cập nhật DB
                │
                ▼  asyncio.Queue (không block pipeline)
          telegram_worker() → gửi ảnh + thông tin lên Telegram
```

---

## Các quyết định kỹ thuật quan trọng

### 1. Pipeline nhận diện khuôn mặt
- **Model:** InsightFace `buffalo_l` — RetinaFace detector + ArcFace R50 (train trên WebFace600K)
- **Embedding:** Vector 512 chiều, L2-normalize, tìm kiếm bằng cosine similarity
- **Lưu trữ DB:** Giữ nguyên từng embedding thô (không mean pooling) → argmax cosine → nhận diện đúng hơn khi góc thay đổi
- **Ngưỡng:** `SIM_THRESHOLD=0.45` được chỉnh trên tập test thực tế

### 2. Ước lượng góc đầu (Head Pose Estimation)
- **Thuật toán:** `cv2.solvePnP` với EPnP trên 5 landmarks của RetinaFace
- **Đầu ra:** Góc Euler Yaw/Pitch/Roll (quy ước ZYX, phân tích từ ma trận Rodrigues)
- **Vấn đề thực tế:** Pitch offset ~+54° do webcam đặt trên màn hình nhìn xuống → cần hiệu chỉnh offset trước khi phân loại pose
- **Ứng dụng:** Hướng dẫn nhân viên chụp đủ 5 tư thế khi đăng ký

### 3. Đăng ký khuôn mặt đa tư thế
- Chụp 11 ảnh: frontal×5, trái×2, phải×2, ngẩng×1, cúi×1
- Kiểm tra chất lượng mỗi frame: độ nét (Laplacian variance ≥60), độ sáng (40–220), tỉ lệ mặt/frame
- Tự động chụp khi cả quality lẫn pose đều đạt yêu cầu

### 4. Logic điểm danh
- **Consecutive confirmation:** Cần 5 frame liên tiếp nhận ra cùng 1 người → lọc false positive từ nhiễu detection
- **Check-in:** `INSERT OR IGNORE` (atomic, mỗi ngày 1 lần duy nhất)
- **Check-out:** `UPDATE` với cooldown 5 phút (last-write-wins pattern)
- **Thread safety:** Camera thread (sync) đẩy event vào asyncio queue qua `run_coroutine_threadsafe`

### 5. Thiết kế database
```
employees.db    → master data (app đăng ký quản lý)
attendance.db   → dữ liệu vận hành (app điểm danh quản lý)

attendance_log:  1 hàng per người per ngày
  UNIQUE(employee_id, date) → đảm bảo không trùng check-in
  check_in  → ghi 1 lần, không đổi
  check_out → cập nhật liên tục, giá trị cuối = giờ ra thực tế

recognition_events: raw log mọi lần nhận diện
  → dùng để debug, tune threshold, phân tích FAR/FRR
```

---

## Kết quả đánh giá

Test trên tập dataset nội bộ (1 nhân viên, 4 kịch bản):

| Kịch bản | Số frame | Kết quả | Độ tin cậy |
|---|---|---|---|
| Đi bình thường | 11 | ✓ Đúng | 91% |
| Đi nhanh | 7 | ✓ Đúng | 71% |
| Quay đầu | 5 | ✓ Đúng | 60% |
| Đeo khẩu trang | 4 | ✗ Unknown | 75% |

```
Accuracy (event-level)       : 3/4 = 75%
FRR (từ chối nhân viên thật) : 25%  ← trường hợp đeo khẩu trang
FAR (nhận nhầm người lạ)     : 0%   ← chưa có dữ liệu unknown
```

**Phân tích lỗi occlusion:** ArcFace R50 không được train trên ảnh có che mặt → embedding lệch xa prototype vector khi đeo khẩu trang. Giải pháp: thêm ảnh đăng ký khi đeo khẩu trang vào DB.

---

## Tech stack

| Thành phần | Công nghệ |
|---|---|
| Phát hiện khuôn mặt | RetinaFace (InsightFace buffalo_l) |
| Nhận diện khuôn mặt | ArcFace R50, train trên WebFace600K |
| Ước lượng góc đầu | OpenCV solvePnP (EPnP algorithm) |
| Backend | FastAPI + Uvicorn |
| Database | SQLite (WAL mode) |
| Async messaging | asyncio.Queue + aiohttp |
| Thông báo | Telegram Bot API (sendPhoto) |
| Camera | OpenCV VideoCapture (RTSP/FFMPEG) |
| ML Runtime | ONNX Runtime (CUDA) |

---

## Cấu trúc project

```
face-attendance-mvp/
├── main.py                        # FastAPI app — điểm vào chính
├── attendance_system/
│   ├── attendance_db.py           # SQLite schema + CRUD
│   ├── attendance_service.py      # Logic check-in/check-out
│   └── telegram_notifier.py      # Async Telegram worker
├── ARCHITECTURE.md                # Thiết kế hệ thống chi tiết
└── .env.example                   # Mẫu biến môi trường

face_registration/
├── main.py                        # App đăng ký khuôn mặt
├── pipeline.py                    # Detect + pose + quality + embed
├── database.py                    # SQLite + FAISS
└── static/register.html          # Giao diện đăng ký
```

---

## Cài đặt và chạy

```bash
# 1. Cài dependencies
pip install fastapi uvicorn insightface opencv-python aiohttp scipy

# 2. Cấu hình môi trường
cp .env.example .env
# Chỉnh .env: RTSP_URL, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# 3. Đăng ký khuôn mặt nhân viên
cd face_registration && python main.py
# Mở http://localhost:8001/register

# 4. Chạy hệ thống điểm danh
cd face-attendance-mvp
source .env && python main.py
# Mở http://localhost:8501/video
```

---

## Biến môi trường

| Biến | Mô tả | Mặc định |
|---|---|---|
| `RTSP_URL` | URL camera RTSP | — |
| `TELEGRAM_BOT_TOKEN` | Token bot từ @BotFather | — |
| `TELEGRAM_CHAT_ID` | ID nhóm Telegram (số âm) | — |
| `SIM_THRESHOLD` | Ngưỡng cosine similarity | 0.45 |
| `DET_THRESHOLD` | Ngưỡng detection confidence | 0.6 |
| `CHECKOUT_COOLDOWN` | Giây giữa 2 lần cập nhật check-out | 300 |

---

## API

```
GET  /video               MJPEG stream nhận diện realtime
GET  /roi/select          Giao diện chọn vùng ROI
GET  /attendance/today    Danh sách điểm danh hôm nay (JSON)
GET  /attendance/{date}   Điểm danh theo ngày (yyyy-mm-dd)
GET  /health              Trạng thái hệ thống
POST /reload              Reload embeddings từ DB
```

---

## Những gì học được qua dự án này

- Xây dựng pipeline computer vision end-to-end: từ camera stream → detection → recognition → database → notification
- Bài toán Head Pose Estimation — inverse projection 2D→3D với `solvePnP`, xử lý gimbal lock và camera offset thực tế
- Production ML considerations: threshold tuning, FAR/FRR tradeoff, embedding drift khi điều kiện thay đổi
- Real-time system design: threading model, asyncio queue, cân bằng latency/throughput (camera 25FPS vs recognition 10FPS)
- SQLite design patterns: atomic INSERT OR IGNORE, WAL mode, FOREIGN KEY constraints, index optimization
- Async/sync boundary: `run_coroutine_threadsafe` để giao tiếp an toàn giữa camera thread và asyncio event loop