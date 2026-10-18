# Upgrade checklist: v0.3.0 to v0.4.0

تضيف هذه النسخة تسجيل المرسلين الجدد من لوحة الإدارة. محاولة C-STORE الأولى من مرسل مجهول تُسجل وتُرفض دون تخزين الصورة. بعد الضغط على Register وحفظ بيانات المركز، يجب على الفرع إعادة الإرسال.

```bash
cd /home/mri
docker compose stop
tar -xzf MRI-03T-CPU-Enhancer-v0.4.0.tar.gz
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer orthanc
docker compose ps
curl -fsS http://127.0.0.1:8000/health
```

اترك `AUTO_PROCESS_ENABLED=false` أثناء التسجيل والاختبار. لا تجعل DICOM متاحًا لكل الإنترنت؛ اسمح في Firewall فقط بعناوين الفروع المعروفة أو استخدم VPN.
