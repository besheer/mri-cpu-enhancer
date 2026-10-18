# Upgrade checklist: v0.4.0 to v0.5.0

تضيف هذه النسخة صفحة الحالات والمقارنة داخل لوحة الإدارة. زر Process BASE ينشئ سلسلة مشتقة باسم `ENHANC-BASE-*` داخل Orthanc فقط ولا يرسلها إلى الفرع.

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v0.5.0.tar.gz
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
docker compose ps
curl -fsS http://127.0.0.1:8000/health
```

افتح `/admin` ثم اضغط «الحالات والمقارنة في OHIF». اترك `AUTO_PROCESS_ENABLED=false` أثناء التقييم.
