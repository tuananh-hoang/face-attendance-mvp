"""
telegram_notifier.py
Async Telegram notifier với retry + rate limit handling.

Setup:
  1. Tạo bot: chat với @BotFather → /newbot → lấy BOT_TOKEN
  2. Thêm bot vào group
  3. Lấy CHAT_ID: https://api.telegram.org/bot{TOKEN}/getUpdates
  4. Set env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Message format:
  🟢 CHECK-IN
  👤 Hoàng Tuấn Anh | Dev
  🕐 08:32:15 — 16/03/2024
  🎯 Score: 0.87
  [ảnh]
"""

import os
import asyncio
import aiohttp
from datetime import datetime

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID",   "")

# Telegram rate limit: 30 msg/s tới 1 group
# Dùng delay 0.1s giữa các message để an toàn
MSG_DELAY = 0.15   # giây


def _format_message(event: dict) -> str:
    emoji    = "🟢" if event["type"] == "check_in" else "🔴"
    label    = "CHECK-IN" if event["type"] == "check_in" else "CHECK-OUT"
    score_pct = f"{event['score']:.0%}"

    return (
        f"{emoji} *{label}*\n"
        f"👤 *{event['name']}*\n"
        f"🕐 {event['time']} — {event['date']}\n"
        f"🎯 Score: {score_pct}"
    )


async def send_event(session: aiohttp.ClientSession, event: dict, retries: int = 3):
    """
    Gửi 1 event lên Telegram.
    Nếu có ảnh → sendPhoto, không có → sendMessage.
    Retry với exponential backoff khi lỗi mạng.
    """
    if not BOT_TOKEN or not CHAT_ID:
        print("[TELEGRAM] BOT_TOKEN hoặc CHAT_ID chưa set — bỏ qua")
        return

    caption = _format_message(event)
    photo_path = event.get("photo_path")

    for attempt in range(retries):
        try:
            if photo_path and os.path.exists(photo_path):
                # Gửi ảnh kèm caption
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
                with open(photo_path, "rb") as f:
                    form = aiohttp.FormData()
                    form.add_field("chat_id",    CHAT_ID)
                    form.add_field("caption",    caption)
                    form.add_field("parse_mode", "Markdown")
                    form.add_field("photo", f, filename="photo.jpg",
                                   content_type="image/jpeg")
                    async with session.post(url, data=form, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data = await resp.json()
                        if not data.get("ok"):
                            raise ValueError(f"Telegram error: {data}")
            else:
                # Không có ảnh → sendMessage
                url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
                payload = {
                    "chat_id"   : CHAT_ID,
                    "text"      : caption,
                    "parse_mode": "Markdown",
                }
                async with session.post(url, json=payload,
                                        timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    if not data.get("ok"):
                        raise ValueError(f"Telegram error: {data}")

            print(f"[TELEGRAM] Sent {event['type']}: {event['name']}")
            return  # Thành công

        except Exception as e:
            wait = 2 ** attempt   # 1s, 2s, 4s
            print(f"[TELEGRAM] Attempt {attempt+1}/{retries} failed: {e}. Retry in {wait}s")
            if attempt < retries - 1:
                await asyncio.sleep(wait)

    print(f"[TELEGRAM] Failed after {retries} retries: {event['name']}")


async def telegram_worker(notify_queue: asyncio.Queue):
    """
    Async worker đọc từ notify_queue và gửi Telegram.
    Chạy song song với recognition pipeline — không block camera.
    """
    print("[TELEGRAM] Worker started")
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                event = await asyncio.wait_for(notify_queue.get(), timeout=1.0)
                await send_event(session, event)
                await asyncio.sleep(MSG_DELAY)  # rate limit
                notify_queue.task_done()
            except asyncio.TimeoutError:
                continue  # không có event, tiếp tục chờ
            except asyncio.CancelledError:
                print("[TELEGRAM] Worker stopped")
                break
            except Exception as e:
                print(f"[TELEGRAM] Worker error: {e}")