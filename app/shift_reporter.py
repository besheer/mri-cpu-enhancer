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

def run_shift_report(shift_type: str | None = None, force: bool = False) -> bool:
    now = datetime.datetime.now()
    weekday = now.weekday()
    hour = now.hour

    # Saturday to Thursday: 5, 6, 0, 1, 2, 3 (4 is Friday)
    if not force:
        if weekday == 4:
            print(f"Skipping: Today is Friday ({AR_DAYS.get(weekday)}), outside work schedule.")
            return False

    day_name = AR_DAYS.get(weekday, "")
    time_str = now.strftime("%I:%M %p").replace("AM", "صباحاً").replace("PM", "مساءً")
    date_str = f"{day_name} {now.strftime('%d/%m/%Y')}"

    # Determine shift type if not explicitly passed
    if not shift_type:
        if hour < 12:
            shift_type = "morning"
        elif hour < 19:
            shift_type = "midday"
        else:
            shift_type = "evening"

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
    recent_series = []
    with connect() as db:
        today_in = db.execute("SELECT count(*) FROM inbound_series WHERE received_at >= date('now', 'localtime')").fetchone()[0]
        today_sent = db.execute("SELECT count(*) FROM return_jobs WHERE status='sent' AND updated_at >= date('now', 'localtime')").fetchone()[0]
        last_row = db.execute("SELECT description, received_at FROM inbound_series ORDER BY received_at DESC LIMIT 1").fetchone()
        if last_row and last_row["description"]:
            last_desc = str(last_row["description"])
        rows = db.execute("SELECT description, instances FROM inbound_series WHERE received_at >= date('now', 'localtime') ORDER BY received_at DESC LIMIT 5").fetchall()
        for r in rows:
            recent_series.append(f"• `{r['description']}` ({r['instances']} شريحة)")

    series_summary = "\n".join(recent_series) if recent_series else "• لم يتم استلام فحوصات جديدة اليوم حتى الآن."

    if shift_type == "morning":
        msg = (
            "🌅 **تقرير بداية الوردية — جاهزية سيرفر الرنين (MRI AI)**\n"
            f"⏰ **التوقيت**: {date_str} — {time_str}\n\n"
            f"• **محطة GE Advantage Workstation**: {ge_status}\n"
            f"• **محرك الذكاء الاصطناعي**: {api_status}\n"
            "• **نظام المعالجة التلقائية**: مفعل بنظام **النسخ الثلاثية** (`SPINE_PRO` / `3T_HD` / `FAST_MRI`)\n"
            "• **حالة الجاهزية**: السيرفر مستعد تماماً وبانتظار أولى حالات اليوم من الفني."
        )
    elif shift_type == "midday":
        msg = (
            "☀️ **تقرير منتصف الوردية — متابعة فحوصات الرنين**\n"
            f"⏰ **التوقيت**: {date_str} — {time_str}\n\n"
            f"• **الفحوصات المستلمة اليوم**: `{today_in}` فحص\n"
            f"• **النسخ المحسنة المرسلة لمحطة العمل**: `{today_sent}` فحص\n"
            f"• **آخر فحص معالج**: `{last_desc}`\n"
            f"• **اتصال محطة GE AW (`adw`)**: {ge_status}\n"
            "• **ملاحظات**: السيرفر مستمر في العمل بكامل طاقته للفترة المسائية."
        )
    else: # evening
        msg = (
            "🌙 **تقرير ختام الوردية — ملخص إنجازات اليوم وحالة السيرفر**\n"
            f"⏰ **التوقيت**: {date_str} — {time_str}\n\n"
            f"• **إجمالي الفحوصات المستلمة اليوم**: `{today_in}` فحص\n"
            f"• **إجمالي النسخ المحسنة المكتملة**: `{today_sent}` فحص تم تسليمها لمحطة GE بنجاح\n"
            f"• **فحوصات اليوم**:\n{series_summary}\n\n"
            f"• **حالة النظام والاتصال**: {api_status} | {ge_status}\n"
            "• **الاستعداد الليلي**: السيرفر يظل في وضع الاستعداد التلقائي لأي حالات طوارئ."
        )

    markup = {
        "inline_keyboard": [
            [{"text": "⚡ تنفيذ آخر تعديلات مطلوبة", "callback_data": "exec_latest_modifications"}],
            [{"text": "📊 فحص فوري للحالة", "callback_data": "status"}],
        ]
    }

    sent = send_telegram_notification(msg, reply_markup=markup)
    print(f"Shift report ({shift_type}) sent successfully to {sent} authorized users at {time_str}.")
    return True

if __name__ == "__main__":
    st = None
    force_run = "--force" in sys.argv
    for arg in sys.argv:
        if arg in ("morning", "midday", "evening"):
            st = arg
    run_shift_report(shift_type=st, force=force_run)
