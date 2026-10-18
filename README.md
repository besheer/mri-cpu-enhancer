# MRI 0.3T CPU Enhancer v1.5.0 — Research MVP

## Synthetic MRI Lab

Open `/admin/synthetic` to generate a reference phantom and controlled Rician noise,
bias-field, motion, ghosting, blur, low-contrast, mixed 0.3T, or extreme artifacts.
Selected enhancement profiles are executed against the same corrupted input and are
stored with the reference in one Orthanc study for OHIF comparison. The lab reports
PSNR, global SSIM, MAE, normalized RMSE, processing time, noise change, and edge
preservation. Every generated DICOM object is marked synthetic/research-only and the
lab never creates a return job.

Version 1.4 adds foreground/ROI and slice-wise metrics, ground-truth edge similarity,
visual difference-map DICOM series, automatic ranking and decisions, CSV export, and
an optional all-level benchmark. Processing profiles now have an independent
`auto_eligible` safety flag. Baseline, CLAHE, unavailable AI/DWI tools, and STRONG
levels cannot be used by branch automation; clinical auto-return still requires QC PASS.

Version 1.4.1 makes the E2E suite self-contained: it uses the automation-eligible
edge-preserving profile, waits for Compose DNS/TCP/DICOM readiness, records PASS/FAIL
in the audit log, and always stops and removes the temporary mock PACS container.

Version 1.5 adds a Protocol Acceleration Study registry at `/admin/protocols`.
It pairs original Standard and Fast series from the same examination, calculates
scan-time reduction, runs comparison-only enhancement jobs, opens all results in
OHIF, stores immutable medical reviews, and exports the registry as CSV. A database
trigger blocks every Protocol Study result from entering the DICOM Return Queue.

نسخة بحثية أولية لتحسين صور MRI منخفضة المجال على CPU مع الحفاظ على ملفات DICOM الأصلية.

> **تحذير:** هذا البرنامج غير معتمد للتشخيص الطبي. الناتج مشتق وتجريبي ويجب عرضه بجانب الأصل ومراجعته طبيًا.

## ما الموجود الآن؟

- API تستقبل DICOM وتعيد Series مشتقة داخل ZIP.
- تحسين محافظ مرجعي (تصحيح bias مبسط + تقليل noise خفيف).
- إنشاء Series/SOP UIDs جديدة وعدم تعديل الملفات الأصلية.
- تسمية الناتج `ENHANC-<Original Series Description>` مع وسم Derived/Research.
- Orthanc عبر Docker Compose لاستقبال صور الجهاز أو PACS.
- Dashboard عربية على `/admin` لإضافة وتعديل وتعطيل وحذف المراكز والأجهزة.
- Routing Rule مستقلة لكل Calling AE: Source IP/AE إلى Return IP/AE/Port.
- Queue لعرض الحالات المرسلة والفاشلة والمتخطاة وآخر خطأ وإعادة المحاولة.
- ملف المركز يبقى `BASE_ONLY` لمسار الإرسال الآلي؛ اختيار المحسن التفصيلي يتم بأمان لكل Series من صفحة الحالات.
- Sender Registration: أول C-STORE من IP/AE مجهول يُرفض ويظهر في Pending Senders، ثم يسجله المدير بنقرة واحدة ويطلب إعادة الإرسال.
- صفحة `/admin/studies` لاختيار الدراسة والسلسلة، وإنشاء `ENHANC-BASE-*` داخل Orthanc للمقارنة فقط، ثم فتح الدراسة في OHIF دون إرسال النتيجة للفرع.
- Processing Queue دائمة مع Profiles: BASE_MILD وDENOISE_MEDIUM وN4_BIAS وN4_DENOISE وCLAHE_VIEW.
- Metrics لكل Job: الزمن والتغير النسبي ومتوسط الضوضاء قبل/بعد، ثم اعتماد يدوي دون إرسال تلقائي.
- صفحة `/admin/tools` للتحكم في تفعيل الأدوات، مع توصيات وتحذيرات لكل Series وحماية Server-side للاختيارات غير المناسبة.
- قبول C-STORE فقط عند تطابق Source IP وCalling AE وCalled AE ونوع MR.
- Queue دائمة تنتظر Stable Series، تمنع التكرار، ثم تعيد الناتج تلقائيًا.
- اختبارات آلية بملفات DICOM صناعية فقط.

المحرك الحالي ليس موديل ذكاء اصطناعي مدربًا؛ هو baseline آمن نسبيًا لاختبار خط DICOM والأداء. بعد استلام بيانات مجهولة الهوية سنضيف ONNX model منفصلًا ونقارن الناتج بالـbaseline.

## تشغيل سريع على CPU

```bash
cp .env.example .env
## عدّل جميع كلمات المرور والـtoken قبل التشغيل
docker compose up --build
```

- API: `http://localhost:8000/docs`
- Admin: `http://localhost:8000/admin`
- Orthanc: `http://localhost:8042` (غيّر بيانات الدخول في `.env`)
- DICOM C-STORE: المنفذ `4242` وAE Title هو `MRI03T`

إرسال ملفات إلى API:

```bash
curl -X POST http://localhost:8000/v1/enhance \
  -F "files=@IM0001.dcm" \
  -F "files=@IM0002.dcm" \
  --output enhanced-series.zip
```

## تشغيل الاختبارات محليًا

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## حدود النسخة الأولى

- تقبل صور MR أحادية الإطار، grayscale، وغير مضغوطة.
- ترفض modalities الأخرى والملفات التي لا تحتوي Pixel Data.
- لا تعالج enhanced multi-frame DICOM أو compressed transfer syntaxes بعد.
- التشغيل التلقائي مغلق افتراضيًا. لا تغيّر `AUTO_PROCESS_ENABLED=true` قبل نجاح C-ECHO واختبار صورة صناعية ذهابًا وعودة.
- اربط `DICOM_BIND_IP` بعنوان WireGuard/Private. لا تستخدم `0.0.0.0` على سيرفر Public.

## الترقية من v0.1.0

خذ نسخة احتياطية من المشروع وملف البيئة، ثم فك الحزمة الجديدة فوق `/home/mri`. احتفظ بكلمات المرور القديمة، وأضف المتغيرات الجديدة من `.env.example`. اجعل `ADMIN_USERNAME` و`ADMIN_PASSWORD` مطابقين لحساب Nginx Basic Auth لتجنب مصادقتين مختلفتين.

بعد تشغيل v0.6.0 افتح `/admin`، وسجّل المرسل من Pending Senders أو أضفه يدويًا، واترك Auto Processing مغلقًا أثناء المقارنة.

لا تفعّل `AUTO_PROCESS_ENABLED` أثناء المقارنة اليدوية. اسم أي سلسلة مشتقة يبدأ دائمًا بـ`ENHANC-`، والسلسلة الأصلية لا تُحذف.

## بيانات المعايرة المطلوبة

عينة أولية من 5–10 فحوص DICOM أصلية مجهولة الهوية، تشمل Brain T2/FLAIR إن أمكن، مع اسم الشركة والموديل وقوة المجال. لا ترفع بيانات مرضى تحتوي اسمًا أو رقمًا أو تاريخ ميلاد.
