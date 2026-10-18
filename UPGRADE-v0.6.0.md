# Upgrade v0.6.0

تضيف النسخة اختيار المحسن لكل Series، وQueue دائمة، وMetrics، واعتمادًا يدويًا. النتائج تبقى داخل Orthanc ولا تُرسل للفرع.

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v0.6.0.tar.gz
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
curl --retry 10 --retry-delay 3 --retry-connrefused --fail http://127.0.0.1:8000/health
```

ابدأ بـBASE_MILD أو DENOISE_MEDIUM على Series واحدة. N4 يعمل على Volume مرتب من ثلاث شرائح متناسقة على الأقل. CLAHE_VIEW للعرض البحثي فقط.
