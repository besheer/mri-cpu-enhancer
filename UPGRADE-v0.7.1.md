# Upgrade v0.7.1

تحديث واجهة الاختيار الجماعي للسلاسل بدون إعادة تحميل الصفحة أو القفز إلى أعلى:

- Checkbox لكل Series أصلية.
- تحديد كل السلاسل الأصلية داخل دراسة واحدة.
- شريط ثابت لاختيار Profile وإضافة المحدد إلى Queue دفعة واحدة.
- التحقق من جميع السلاسل على السيرفر قبل إنشاء أي Job.
- منع إعادة معالجة سلاسل `ENHANC-*`.
- توضيح متطلبات MONAI وDIPY في صفحة إدارة الأدوات.

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v0.7.1.tar.gz
grep -n 'version=' app/main.py
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
curl --retry 15 --retry-delay 3 --retry-connrefused --fail http://127.0.0.1:8000/health
```

يجب أن يعرض `/openapi.json` الإصدار `0.7.1`.
