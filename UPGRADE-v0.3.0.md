# Upgrade checklist: v0.2.0 to v0.3.0

1. احتفظ بالنسخة الاحتياطية الحالية ولا تحذف Docker volumes.
2. اجعل `AUTO_PROCESS_ENABLED=false` قبل الترقية.
3. فك الحزمة فوق `/home/mri` مع الاحتفاظ بملف `.env` الحالي.
4. شغّل `docker compose config -q` ثم ابنِ خدمة enhancer من جديد.
5. افتح `/admin`. ترقية SQLite تتم تلقائيًا وتُبقي المراكز القديمة.
6. أكمل بيانات كل مركز واختر `BASE_ONLY` فقط.
7. نفّذ C-ECHO واختبار DICOM صناعي قبل فتح WAN أو تشغيل المعالجة الآلية.

```bash
cd /home/mri
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
docker compose ps
curl -fsS http://127.0.0.1:8000/health
```

لا تجعل `DICOM_BIND_IP=0.0.0.0`. استخدم عنوان VPN/واجهة خاصة، أو Public IP مع Firewall allowlist صارمة للمراكز المسجلة فقط.
