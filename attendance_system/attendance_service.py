"""
attendance_service.py
Business logic check-in / check-out.

Flow:
  Track exit (ByteTrack callback)
    → recognize best frame của track
    → quyết định check_in hoặc check_out
    → lưu ảnh
    → ghi DB
    → đẩy vào notify_queue
"""

import os
import asyncio
import threading
from datetime import datetime, date
from typing import Optional

import cv2
import numpy as np

from attendance_db import (
    has_checkin_today,
    record_checkin,
    record_checkout,
    log_recognition_event,
)

# ── Config ──
PHOTOS_DIR        = os.environ.get("PHOTOS_DIR", "data/photos")
CHECKOUT_COOLDOWN = int(os.environ.get("CHECKOUT_COOLDOWN", 300))  # 5 phút
SIM_THRESHOLD     = float(os.environ.get("SIM_THRESHOLD", 0.45))

# ── State ──
# last_checkout[employee_id] = timestamp — throttle checkout update
_last_checkout : dict = {}
_state_lock = threading.Lock()

# Queue async để gửi Telegram (non-blocking)
notify_queue: asyncio.Queue = None   # set từ bên ngoài khi khởi động



def _process_frame_inline(frame):
    """Detect + embed trực tiếp, tránh conflict với package pipeline."""
    try:
        import sys
        main_mod = sys.modules.get("__main__") or sys.modules.get("main")
        if main_mod is None:
            return {"status": "no_main"}
        face_app = getattr(main_mod, "face_app", None)
        if face_app is None:
            return {"status": "no_face_app"}
        faces = face_app.get(frame)
        if not faces:
            return {"status": "no_face"}
        face = max(faces, key=lambda f: f.det_score)
        if face.det_score < 0.5:
            return {"status": "low_confidence", "det_score": float(face.det_score)}
        return {
            "status"   : "ok",
            "det_score": float(face.det_score),
            "embedding": face.normed_embedding.tolist(),
        }
    except Exception as e:
        print(f"[SERVICE] inline error: {e}")
        return {"status": "error"}

def set_notify_queue(q: asyncio.Queue):
    global notify_queue, _event_loop
    notify_queue = q
    try:
        _event_loop = asyncio.get_event_loop()
    except RuntimeError:
        _event_loop = None
    print(f"[SERVICE] notify_queue set, event_loop: {_event_loop}")


# ════════════════════════════════════════════
#  Lưu ảnh
# ════════════════════════════════════════════

def save_photo(frame: np.ndarray, employee_id: str, event_type: str) -> str:
    """
    Lưu frame vào data/photos/{date}/{employee_id}_{event_type}_{HHMMss}.jpg
    Trả về đường dẫn tương đối.
    """
    today     = date.today().isoformat()
    ts        = datetime.now().strftime("%H%M%S")
    save_dir  = os.path.join(PHOTOS_DIR, today)
    os.makedirs(save_dir, exist_ok=True)

    filename  = f"{employee_id}_{event_type}_{ts}.jpg"
    full_path = os.path.join(save_dir, filename)
    cv2.imwrite(full_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return full_path


# ════════════════════════════════════════════
#  Process track exit (ByteTrack callback)
# ════════════════════════════════════════════

def on_track_exit(track, db_embs, db_names, recognize_fn):
    """
    Gọi khi ByteTrack xác định track đã exit frame.
    track.best_frame: frame tốt nhất của track (det_score cao nhất)
    track.best_score: det_score cao nhất

    Chạy trong camera thread — nhanh, không block.
    Đẩy notify vào queue async.
    """
    if track.best_frame is None:
        return
    if len(db_embs) == 0:
        return

    # ── Recognize best frame ──
    result = _process_frame_inline(track.best_frame)

    if result["status"] != "ok" or result["embedding"] is None:
        log_recognition_event(
            employee_id=None, name="Unknown",
            score=0.0, event_type="unknown",
            track_id=track.track_id,
        )
        return

    emb          = np.array(result["embedding"], dtype=np.float32)
    name, score  = recognize_fn(emb)

    if name == "Unknown" or score < SIM_THRESHOLD:
        log_recognition_event(
            employee_id=None, name="Unknown",
            score=float(score), event_type="unknown",
            track_id=track.track_id,
            det_score=result.get("det_score"),
        )
        return

    # Lấy employee_id thật từ SQLite dựa trên display name
    import sqlite3 as _sqlite3
    employees_db = os.environ.get("EMPLOYEES_DB", "")
    employee_id  = None
    if os.path.exists(employees_db):
        try:
            _conn = _sqlite3.connect(employees_db)
            row   = _conn.execute(
                "SELECT employee_id FROM employees WHERE name = ? LIMIT 1",
                (name,)
            ).fetchone()
            _conn.close()
            if row:
                employee_id = row[0]
        except Exception as e:
            print(f"[SERVICE] lookup employee_id error: {e}")

    if employee_id is None:
        # Fallback: dùng name slug nếu không tìm thấy
        employee_id = name.lower().replace(" ", "_")

    _process_event(
        employee_id = employee_id,
        name        = name,
        score       = score,
        frame       = track.best_frame,
        track_id    = track.track_id,
        det_score   = result.get("det_score"),
    )


def _process_event(
    employee_id : str,
    name        : str,
    score       : float,
    frame       : np.ndarray,
    track_id    : int   = None,
    det_score   : float = None,
):
    """
    Quyết định check_in hay check_out, ghi DB, lưu ảnh, push notify.
    """
    now   = datetime.now()
    today = now.date().isoformat()

    with _state_lock:
        already_in = has_checkin_today(employee_id, today)

        if not already_in:
            # ── CHECK-IN ──
            photo_path = save_photo(frame, employee_id, "checkin")
            record_checkin(employee_id, name, score, photo_path, now)
            log_recognition_event(
                employee_id, name, score, "check_in",
                photo_path, track_id, det_score, now
            )
            print(f"[CHECKIN]  {name:20s} score={score:.3f} track={track_id}")

            if notify_queue is not None:
                _push_notify({
                    "type"       : "check_in",
                    "name"       : name,
                    "employee_id": employee_id,
                    "score"      : score,
                    "time"       : now.strftime("%H:%M:%S"),
                    "date"       : now.strftime("%d/%m/%Y"),
                    "photo_path" : photo_path,
                })

        else:
            # ── CHECK-OUT ──
            last = _last_checkout.get(employee_id)
            if last and (now - last).total_seconds() < CHECKOUT_COOLDOWN:
                # Cooldown chưa qua — chỉ log, không update
                log_recognition_event(
                    employee_id, name, score, "ignored",
                    None, track_id, det_score, now
                )
                return

            photo_path = save_photo(frame, employee_id, "checkout")
            record_checkout(employee_id, name, score, photo_path, now)
            _last_checkout[employee_id] = now
            log_recognition_event(
                employee_id, name, score, "check_out",
                photo_path, track_id, det_score, now
            )
            print(f"[CHECKOUT] {name:20s} score={score:.3f} track={track_id}")

            if notify_queue is not None:
                _push_notify({
                    "type"       : "check_out",
                    "name"       : name,
                    "employee_id": employee_id,
                    "score"      : score,
                    "time"       : now.strftime("%H:%M:%S"),
                    "date"       : now.strftime("%d/%m/%Y"),
                    "photo_path" : photo_path,
                })


def _push_notify(event: dict):
    """Push vào asyncio queue từ sync thread (thread-safe)."""
    if notify_queue is None:
        return
    try:
        if _event_loop is not None and _event_loop.is_running():
            # Gọi từ worker thread → dùng run_coroutine_threadsafe
            asyncio.run_coroutine_threadsafe(notify_queue.put(event), _event_loop)
        else:
            # Fallback nếu không có loop
            notify_queue.put_nowait(event)
    except asyncio.QueueFull:
        print("[NOTIFY] Queue full — dropped event")
    except Exception as e:
        print(f"[NOTIFY] push failed: {e}")