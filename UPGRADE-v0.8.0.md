# Upgrade v0.8.0

يضيف الإصدار مسارين جديدين للمقارنة البحثية:

- `EDGE_DENOISE_MILD`: تقليل ضوضاء باستخدام Anisotropic Diffusion مع مزج محافظ.
- `N4_EDGE_DENOISE`: تصحيح N4 ثم تقليل ضوضاء محافظ للحواف.

ويضيف:

- مستويات `MILD` و`MEDIUM` و`STRONG` للمحسنات، مع إبقاء Baseline ثابتًا.
- اشتراط إقرار صريح للمستوى `STRONG` وCLAHE.
- اختيار عدة محسنات وعدة Series وإضافتها إلى Queue دفعة واحدة.
- تسجيل Strength داخل Job وSeries Description وDerivation Description.
- مقياس `edge_preservation_percent` بجوار الضوضاء والزمن.

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v0.8.0.tar.gz
grep -n 'version=' app/main.py
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
curl --retry 15 --retry-delay 3 --retry-connrefused --fail http://127.0.0.1:8000/health
```

يجب أن يعرض `/openapi.json` الإصدار `0.8.0`.
