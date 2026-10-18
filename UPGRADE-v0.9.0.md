# Upgrade v0.9.0

تضيف هذه النسخة طبقة تقييم بحثي منظمة فوق محركات v0.8.0:

- Safe Comparison Preset: Baseline MILD وEdge MILD وEdge MEDIUM.
- Quality Gate بنتائج PASS/WARN/FAIL.
- منع اعتماد نتيجة FAIL، مع بقاء قرار الرفض متاحًا.
- تقييم الطبيب: الأفضل أو مقبول أو مرفوض، مع ملاحظة وتاريخ.
- فلترة Queue حسب الحالة والمحسن.
- Difference Map عند الطلب بصيغة PNG، لا تُحفظ في Orthanc ولا تُرسل للمركز.
- ترحيل تلقائي غير مدمر لقاعدة SQLite الحالية.

## الترقية

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v0.9.0.tar.gz
grep -n 'version=' app/main.py
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
curl --retry 15 --retry-delay 3 --retry-connrefused --fail http://127.0.0.1:8000/health
```

يجب أن يعرض `/openapi.json` الإصدار `0.9.0`. لا يغيّر التحديث إعدادات WAN أو Orthanc أو الإرسال التلقائي.
