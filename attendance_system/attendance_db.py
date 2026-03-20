"""
attendance_db.py
SQLite schema và CRUD operations cho hệ thống điểm danh.

3 bảng:
  employees         → master data (sync từ face_registration)
  attendance_log    → 1 hàng/người/ngày, check_in + check_out
  recognition_events → raw log mỗi lần nhận diện, dùng debug + threshold tuning
"""

import os
import sqlite3
from datetime import datetime, date
from typing import Optional
from contextlib import contextmanager

# ── Config ──
ATTENDANCE_DB  = os.environ.get("ATTENDANCE_DB", "data/attendance.db")
EMPLOYEES_DB   = os.environ.get("EMPLOYEES_DB",  "data/employees.db")
PHOTOS_DIR     = os.environ.get("PHOTOS_DIR",    "data/photos")


# ════════════════════════════════════════════
#  Connection
# ════════════════════════════════════════════

@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(ATTENDANCE_DB), exist_ok=True)
    conn = sqlite3.connect(ATTENDANCE_DB, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # Write-Ahead Logging — tốt hơn cho concurrent read
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ════════════════════════════════════════════
#  Schema
# ════════════════════════════════════════════

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id  TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    title        TEXT DEFAULT '',
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS attendance_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id     TEXT NOT NULL,
    date            DATE NOT NULL,
    check_in        DATETIME,
    check_out       DATETIME,
    checkin_photo   TEXT,       -- đường dẫn ảnh lúc check-in
    checkout_photo  TEXT,       -- đường dẫn ảnh lúc check-out
    checkin_score   REAL,       -- cosine score cao nhất lúc check-in
    checkout_score  REAL,       -- cosine score cao nhất lúc check-out
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(employee_id, date),
    FOREIGN KEY(employee_id) REFERENCES employees(employee_id)
);

CREATE TABLE IF NOT EXISTS recognition_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id  TEXT,
    name         TEXT,
    timestamp    DATETIME NOT NULL,
    score        REAL NOT NULL,
    event_type   TEXT NOT NULL,  -- check_in / check_out / ignored / unknown
    photo_path   TEXT,
    track_id     INTEGER,
    det_score    REAL,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_attendance_date
    ON attendance_log(date, employee_id);

CREATE INDEX IF NOT EXISTS idx_events_timestamp
    ON recognition_events(timestamp);

CREATE INDEX IF NOT EXISTS idx_events_employee
    ON recognition_events(employee_id, timestamp);
"""


def init_db():
    """Tạo schema nếu chưa có. Gọi 1 lần lúc khởi động."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    print(f"[DB] attendance.db initialized: {ATTENDANCE_DB}")
    _sync_employees()


def _sync_employees():
    """
    Sync employees từ face_registration DB sang attendance DB.
    Gọi lúc khởi động để đảm bảo 2 DB nhất quán.
    """
    if not os.path.exists(EMPLOYEES_DB):
        print(f"[DB] employees.db chưa có tại {EMPLOYEES_DB} — bỏ qua sync")
        return

    src  = sqlite3.connect(EMPLOYEES_DB)
    rows = src.execute("SELECT employee_id, name, title FROM employees").fetchall()
    src.close()

    with get_conn() as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO employees (employee_id, name, title) VALUES (?, ?, ?)",
            rows
        )
    print(f"[DB] Synced {len(rows)} employees từ face_registration DB")


# ════════════════════════════════════════════
#  Attendance CRUD
# ════════════════════════════════════════════

def _ensure_employee(conn, employee_id: str, name: str = ""):
    """Tự động insert employee nếu chưa có — tránh FOREIGN KEY error."""
    conn.execute(
        "INSERT OR IGNORE INTO employees (employee_id, name) VALUES (?, ?)",
        (employee_id, name)
    )


def record_checkin(
    employee_id : str,
    name        : str,
    score       : float,
    photo_path  : str,
    timestamp   : datetime = None,
) -> bool:
    """
    Ghi check-in nếu hôm nay chưa có.
    Trả về True nếu ghi thành công, False nếu đã check-in rồi.
    """
    if timestamp is None:
        timestamp = datetime.now()
    today = timestamp.date().isoformat()
    ts    = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    with get_conn() as conn:
        # Đảm bảo employee tồn tại trước khi insert attendance
        _ensure_employee(conn, employee_id, name)
        # INSERT OR IGNORE — atomic, không cần check trước
        conn.execute("""
            INSERT OR IGNORE INTO attendance_log
                (employee_id, date, check_in, checkin_photo, checkin_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (employee_id, today, ts, photo_path, score, ts))

        # Kiểm tra có ghi không (rowcount=0 nếu đã tồn tại)
        row = conn.execute(
            "SELECT check_in FROM attendance_log WHERE employee_id=? AND date=?",
            (employee_id, today)
        ).fetchone()

    already_checked = row and row["check_in"] != ts
    return not already_checked


def record_checkout(
    employee_id : str,
    name        : str,
    score       : float,
    photo_path  : str,
    timestamp   : datetime = None,
) -> None:
    """
    Cập nhật check-out. Luôn update (last-write wins).
    """
    if timestamp is None:
        timestamp = datetime.now()
    today = timestamp.date().isoformat()
    ts    = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    with get_conn() as conn:
        conn.execute("""
            UPDATE attendance_log
            SET check_out      = ?,
                checkout_photo = ?,
                checkout_score = ?,
                updated_at     = ?
            WHERE employee_id = ? AND date = ?
        """, (ts, photo_path, score, ts, employee_id, today))


def has_checkin_today(employee_id: str, today: str = None) -> bool:
    if today is None:
        today = date.today().isoformat()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM attendance_log WHERE employee_id=? AND date=? AND check_in IS NOT NULL",
            (employee_id, today)
        ).fetchone()
    return row is not None


def log_recognition_event(
    employee_id : Optional[str],
    name        : str,
    score       : float,
    event_type  : str,
    photo_path  : str = None,
    track_id    : int = None,
    det_score   : float = None,
    timestamp   : datetime = None,
):
    if timestamp is None:
        timestamp = datetime.now()
    ts = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO recognition_events
                (employee_id, name, timestamp, score, event_type, photo_path, track_id, det_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (employee_id, name, ts, score, event_type, photo_path, track_id, det_score))


# ════════════════════════════════════════════
#  Query
# ════════════════════════════════════════════

def get_attendance_today() -> list:
    today = date.today().isoformat()
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT e.name, e.title, a.check_in, a.check_out,
                   a.checkin_score, a.checkout_score,
                   a.checkin_photo, a.checkout_photo,
                   a.employee_id
            FROM attendance_log a
            JOIN employees e ON a.employee_id = e.employee_id
            WHERE a.date = ?
            ORDER BY a.check_in ASC
        """, (today,)).fetchall()
    return [dict(r) for r in rows]


def get_attendance_by_date(target_date: str) -> list:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT e.name, e.title, a.check_in, a.check_out,
                   a.checkin_score, a.checkout_score, a.employee_id
            FROM attendance_log a
            JOIN employees e ON a.employee_id = e.employee_id
            WHERE a.date = ?
            ORDER BY a.check_in ASC
        """, (target_date,)).fetchall()
    return [dict(r) for r in rows]


def get_employee_attendance(employee_id: str, days: int = 30) -> list:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT date, check_in, check_out, checkin_score, checkout_score
            FROM attendance_log
            WHERE employee_id = ?
              AND date >= date('now', ? || ' days')
            ORDER BY date DESC
        """, (employee_id, f"-{days}")).fetchall()
    return [dict(r) for r in rows]