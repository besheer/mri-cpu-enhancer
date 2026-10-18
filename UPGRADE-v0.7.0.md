# Upgrade v0.7.0

تضيف هذه النسخة إدارة أدوات التحسين من GUI وتوصيات تلقائية وتحذيرات وقواعد منع للاختيارات غير المناسبة.

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v0.7.0.tar.gz
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
curl --retry 15 --retry-delay 3 --retry-connrefused --fail http://127.0.0.1:8000/health
```

افتح `/admin/studies` ثم «إدارة أدوات التحسين». Baseline محمي ولا يمكن تعطيله. MONAI وDIPY ظاهران كغير جاهزين حتى إضافة نموذج موثق ودعم DWI.
