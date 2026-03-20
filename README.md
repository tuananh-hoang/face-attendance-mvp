# Kiến trúc hệ thống điểm danh khuôn mặt

## Tổng quan

Hệ thống gồm 2 app độc lập, dùng chung database:

```
┌─────────────────────────┐         ┌─────────────────────────┐
│   face_registration/    │         │   face-attendance-mvp/  │
│   App đăng ký (8001)    │  ─────▶  │   App điểm danh (8501) │
│   Nhân viên tự chụp     │  SQLite  │   Camera RTSP cửa ra   │
│   11 tư thế             │  shared  │   Nhận diện realtime   │
└─────────────────────────┘         └─────────────────────────┘
          │                                     │
          ▼                                     ▼
   employees.db                         attendance.db
   face_embeddings                       attendance_log
                                         recognition_events
```

---

## Cấu trúc thư mục

```
/home/anhht/
│
├── face_registration/              ← App đăng ký khuôn mặt
│   ├── main.py                     ← FastAPI server port 8501
│   ├── pipeline.py                 ← detect + pose + quality + embedding
│   ├── database.py                 ← FAISS + SQLite operations
│   ├── models.py                   ← Pydantic schemas
│   ├── static/register.html        ← Web UI đăng ký
│   └── data/
│       ├── employees.db            ← SOURCE OF TRUTH (employees + embeddings)
│       ├── faiss.index             ← vector search index (không dùng trong điểm danh)
│       └── id_map.json             ← FAISS idx → employee_id
│
└── face-attendance-mvp/            ← App điểm danh
    ├── main.py                     ← FastAPI server port 8501 (entry point)
    ├── roi.npy                     ← tọa độ ROI đã chọn [x1,y1,x2,y2]
    ├── .env                        ← credentials (TELEGRAM_BOT_TOKEN, CHAT_ID...)
    ├── attendance_system/          ← modules điểm danh
    │   ├── attendance_db.py        ← SQLite schema + CRUD attendance
    │   ├── attendance_service.py   ← business logic check-in/check-out
    │   └── telegram_notifier.py    ← async gửi Telegram
    └── data/
        ├── attendance.db           ← điểm danh (attendance_log, recognition_events)
        └── photos/
            └── 2024-03-16/
                ├── hoang_tuan_anh_checkin_083215.jpg
                └── hoang_tuan_anh_checkout_174503.jpg
```

---

## Luồng dữ liệu chính

### 1. Đăng ký khuôn mặt (chạy 1 lần per nhân viên)

```
Nhân viên mở browser → face_registration:8001/register
        │
        ▼
Webcam stream → gửi frame lên POST /process_frame mỗi 300ms
        │
        ▼ (server xử lý)
RetinaFace detect → 5 landmarks → solvePnP → yaw/pitch/roll → pose_label
                 → quality check (blur, brightness, face_ratio)
                 → ArcFace → normed_embedding (512,)
        │
        ▼ (trả về client)
Client nhận pose_label → hiển thị hướng dẫn → auto-capture khi đủ điều kiện
        │
        ▼ (sau khi đủ 11 embeddings)
POST /submit → server lưu vào SQLite:
    employees(employee_id, name, title)
    face_embeddings(employee_id, embedding BLOB, pose)
        │
        ▼
Rebuild FAISS index → faiss.index + id_map.json
```

### 2. Điểm danh realtime (chạy 24/7)

```
Camera RTSP 25FPS
        │
        ▼ camera_reader() [thread riêng]
latest_frame (luôn là frame mới nhất)
        │
        ▼ gen_frames() [generator, mỗi 40ms]
Crop ROI → resize 640×480
        │
        ▼
face_app.get() → [face_1, face_2, ...]
        │
        ▼ loop từng face
det_score >= 0.6 và face_w >= 60px?
        │ có
        ▼
recognize(normed_embedding):
    sims = db_embs @ face_emb    ← cosine similarity
    idx  = argmax(sims)
    score >= 0.45 → trả tên
    score <  0.45 → "Unknown"
        │
        ▼
_consec_count[name] += 1
_consec_count[name] >= 5?       ← 5 frame liên tiếp
        │ có
        ▼
_handle_attendance(name, score, frame, bbox)
        │
        ├── Hôm nay chưa check-in? → record_checkin() → notify CHECK-IN
        └── Đã check-in rồi?       → checkout cooldown qua? → record_checkout() → notify CHECK-OUT
                │
                ▼
        save_photo() → data/photos/{date}/{employee_id}_{type}_{time}.jpg
        log_recognition_event() → recognition_events table
        _push_notify() → notify_queue (asyncio.Queue)
                │
                ▼ [telegram_worker, async, riêng biệt]
        sendPhoto API → Telegram Group
```

---

## Kết nối giữa các file

```
main.py
  │
  ├── import InsightFace → face_app (global)
  │       dùng trong: gen_frames(), _handle_attendance()
  │
  ├── import attendance_system/attendance_db.py
  │       dùng: init_db(), has_checkin_today(), record_checkin(),
  │             record_checkout(), log_recognition_event()
  │             get_attendance_today(), get_attendance_by_date()
  │
  ├── import attendance_system/attendance_service.py
  │       dùng: set_notify_queue(), save_photo(), _push_notify()
  │       _handle_attendance() định nghĩa TRONG main.py
  │       (tránh circular import)
  │
  └── import attendance_system/telegram_notifier.py
          dùng: telegram_worker(notify_queue)
                chạy như asyncio.Task song song với uvicorn
```

---

## Database Schema

### employees.db (thuộc face_registration, read-only với app điểm danh)

```sql
employees
  employee_id  TEXT  PK    -- "hoang_tuan_anh_k3f2a"
  name         TEXT        -- "Hoàng Tuấn Anh"
  title        TEXT        -- "Dev"
  created_at   DATETIME

face_embeddings
  id           INTEGER PK
  employee_id  TEXT    FK → employees
  embedding    BLOB        -- numpy float32 (512,) serialized
  pose         TEXT        -- frontal/left/right/up/down
  created_at   DATETIME
```

### attendance.db (thuộc face-attendance-mvp)

```sql
employees                   ← mirror từ face_registration (auto-sync lúc khởi động)
  employee_id  TEXT  PK
  name         TEXT
  title        TEXT

attendance_log              ← 1 hàng per người per ngày
  id              INTEGER PK
  employee_id     TEXT    FK
  date            DATE        -- "2024-03-16"
  check_in        DATETIME    -- ghi 1 lần, không đổi
  check_out       DATETIME    -- UPDATE mỗi lần nhận diện (last-write wins)
  checkin_photo   TEXT        -- đường dẫn file ảnh
  checkout_photo  TEXT
  checkin_score   REAL        -- cosine score lúc check-in
  checkout_score  REAL
  UNIQUE(employee_id, date)

recognition_events          ← raw log mỗi lần nhận diện, dùng debug
  id           INTEGER PK
  employee_id  TEXT
  name         TEXT
  timestamp    DATETIME
  score        REAL
  event_type   TEXT        -- check_in / check_out / ignored / unknown
  photo_path   TEXT
```

---

## Threading và Async model

```
Process main.py
│
├── Thread 1: camera_reader()          [daemon, vòng lặp vô hạn]
│     → đọc RTSP 25FPS
│     → ghi vào latest_frame (protected by frame_lock)
│
├── Thread 2+: uvicorn worker threads  [AnyIO threadpool]
│     → xử lý HTTP request
│     → gen_frames() chạy trong thread này
│         → đọc latest_frame
│         → detect + recognize
│         → _handle_attendance() → _push_notify()
│
└── Async event loop (uvicorn):
      └── Task: telegram_worker()      [asyncio coroutine]
            → đọc notify_queue
            → gửi Telegram API
            → retry với exponential backoff

Giao tiếp giữa thread và async:
  Thread → asyncio.run_coroutine_threadsafe(queue.put(event), event_loop)
  Async  → await queue.get()
```

---

## Check-in / Check-out Logic

```
State trong RAM (reset khi restart):
  _consec_count = {"Hoàng Tuấn Anh": 3}   ← đang đếm
  _last_seen    = {"co_hoang_...": datetime} ← cooldown checkout

Mỗi frame nhận diện được name:
  ┌─ count < 5  → tăng count, bỏ qua
  └─ count >= 5 → reset count về 0
          │
          ▼
          has_checkin_today(employee_id)?
          │
          ├─ Chưa → INSERT check_in (INSERT OR IGNORE → atomic)
          │         gửi Telegram 🟢 CHECK-IN
          │
          └─ Rồi  → (now - last_checkout) >= 5 phút?
                    │
                    ├─ Chưa đủ → bỏ qua (tránh spam checkout)
                    └─ Đủ     → UPDATE check_out
                                 gửi Telegram 🔴 CHECK-OUT
```

---

## API Endpoints

| Endpoint | Method | Mô tả |
|---|---|---|
| `/video` | GET | MJPEG stream nhận diện realtime |
| `/roi/select` | GET | Web UI chọn vùng ROI |
| `/roi/save` | POST | Lưu tọa độ ROI |
| `/roi/current` | GET | Tọa độ ROI hiện tại |
| `/snapshot` | GET | 1 frame JPEG từ camera |
| `/attendance/today` | GET | Danh sách điểm danh hôm nay |
| `/attendance/{date}` | GET | Điểm danh theo ngày (yyyy-mm-dd) |
| `/health` | GET | Trạng thái hệ thống |
| `/reload` | POST | Reload DB embeddings |
| `/record/start` | POST | Bắt đầu ghi video test |
| `/record/stop` | POST | Dừng ghi video |

---

## Telegram Message Format

```
🟢 CHECK-IN
👤 Hoàng Tuấn Anh
🕐 08:32:15 — 16/03/2024
🎯 Score: 87%
[ảnh khuôn mặt lúc check-in]

🔴 CHECK-OUT
👤 Hoàng Tuấn Anh
🕐 17:45:03 — 16/03/2024
🎯 Score: 91%
[ảnh khuôn mặt lúc check-out]
```

---

## Các thông số có thể điều chỉnh trong .env

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `SIM_THRESHOLD` | 0.45 | Ngưỡng cosine để nhận diện |
| `DET_THRESHOLD` | 0.6 | Ngưỡng detection score |
| `MIN_FACE_PX` | 60 | Chiều rộng mặt tối thiểu (px) |
| `CHECKOUT_COOLDOWN` | 300 | Giây giữa 2 lần update check-out |
| `TELEGRAM_BOT_TOKEN` | — | Token bot Telegram |
| `TELEGRAM_CHAT_ID` | — | ID group Telegram |
| `RTSP_URL` | — | URL camera RTSP |
