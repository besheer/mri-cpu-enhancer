import os
import sys
import json
import datetime
import httpx
import sqlite3

sys.path.insert(0, '/app')

from app.store import connect, setting
from app.telegram_bot import send_telegram_notification

AR_DAYS = {
    5: "السبت",
    6: "الأحد",
    0: "الاثنين",
    1: "الثلاثاء",
    2: "الأربعاء",
    3: "الخميس",
    4: "الجمعة",
}

def check_and_send_heartbeat(force: bool = False) -> bool:
    now = datetime.datetime.now()
    weekday = now.weekday()
    hour = now.hour

    # Saturday to Thursday: 5, 6, 0, 1, 2, 3 (4 is Friday)
    # Hours: 9 AM to 11 PM (9 to 23 inclusive)
    if not force:
        if weekday == 4:
            print(f"Skipping: Today is Friday ({AR_DAYS.get(weekday)}), outside work schedule.")
            return False
        if hour < 9 or hour > 23:
            print(f"Skipping: Current hour ({hour}) is outside working hours (09:00 - 23:00).")
            return False

    day_name = AR_DAYS.get(weekday, "")
    time_str = now.strftime("%I:%M %p").replace("AM", "صباحاً").replace("PM", "مساءً")
    date_str = f"{day_name} {now.strftime('%d/%m/%Y')}"

    # 1. Health check of Fast API
    api_status = "يعمل بكفاءة 100%"
    try:
        with httpx.Client(timeout=5) as client:
            r = client.get("http://127.0.0.1:8000/health")
            if r.status_code != 200:
                api_status = f"استجابة غير طبيعية ({r.status_code})"
    except Exception as e:
        api_status = f"تنبيه ({e})"

    # 2. Check connection to GE Advantage Workstation
    ge_status = "متصل بنجاح (200 OK)"
    try:
        with httpx.Client(auth=('admin', 'orthancPassword2026!'), timeout=8) as client:
            client.put('http://orthanc:8042/modalities/adw', json={
                "AET": "adw",
                "Host": "192.168.0.206",
                "Port": 4006
            })
            echo_r = client.post('http://orthanc:8042/modalities/adw/echo')
            if echo_r.status_code != 200:
                ge_status = f"غير مستجيب ({echo_r.status_code})"
    except Exception as e:
        ge_status = f"تنبيه اتصال ({e})"

    # 3. Query cases & stats
    today_in = 0
    today_sent = 0
    last_desc = "لا توجد بعد"
    with connect() as db:
        today_in = db.execute("SELECT count(*) FROM inbound_series WHERE received_at >= date('now', 'localtime')").fetchone()[0]
        today_sent = db.execute("SELECT count(*) FROM return_jobs WHERE status='sent' AND updated_at >= date('now', 'localtime')").fetchone()[0]
        last_row = db.execute("SELECT description, received_at FROM inbound_series ORDER BY received_at DESC LIMIT 1").fetchone()
        if last_row and last_row["description"]:
            last_desc = str(last_row["description"])

    msg = (
        "🟢 **تقرير الجاهزية الدوري لسيرفر الرنين (MRI AI)**\n"
        f"⏰ **التوقيت**: {date_str} — {time_str}\n\n"
        f"• **سيرفر الذكاء الاصطناعي**: {api_status}\n"
        f"• **محطة العمل GE AW (`adw`)**: {ge_status}\n"
        f"• **حالات اليوم الواردة**: `{today_in}` فحص\n"
        f"• **النسخ المرسلة اليوم**: `{today_sent}` فحص\n"
        f"• **آخر فحص معالج**: `{last_desc}`\n"
        "• **حالة النظام**: السيرفر يعمل بكامل جاهزيته، وبانتظار استقبال الحالات الجديدة أو أي تعليمات من الفني."
    )

    markup = {
        "inline_keyboard": [
            [{"text": "⚡ تنفيذ آخر تعديلات مطلوبة", "callback_data": "exec_latest_modifications"}],
            [{"text": "📊 فحص فوري للحالة", "callback_data": "status"}],
        ]
    }

    sent = send_telegram_notification(msg, reply_markup=markup)
    print(f"Heartbeat sent successfully to {sent} authorized users at {time_str}.")
    return True

if __name__ == "__main__":
    force_run = "--force" in sys.argv
    check_and_send_heartbeat(force=force_run)
