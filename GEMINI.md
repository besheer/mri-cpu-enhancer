# MRI AI Appliance & GE Workstation Knowledge Base

## Infrastructure & Network Topology

| Component | IP Address | Ports | Credentials / Protocol | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Local Host** | `192.168.0.208` | - | Windows Server 2016 | Workspace & Agent Host (`OpenSSH` in `C:\Program Files\OpenSSH`) |
| **DICOM / AI Server** | `192.168.0.207` | `22` (SSH)<br>`4242` (DICOM)<br>`8000` (FastAPI)<br>`8042` (Orthanc) | `root` / `mri`<br>SSH key: `~/.ssh/id_ed25519` | Ubuntu Host (`dicom`), Docker runner for NAFNet AI & Orthanc |
| **GE Modality Workstation** | `192.168.0.206` | `4006` (DICOM) | AE Title: `adw` (lowercase) | GE Advantage Workstation (AW/ADW) |

---

## Remote Appliance Directories & Environment

* **Project Path**: `/home/mri/mri-cpu-enhancer`
* **Docker Compose Stack**:
  * `mri-cpu-enhancer-enhancer-1`: FastAPI backend, NAFNet inference engine (`http://192.168.0.207:8000`)
  * `mri-cpu-enhancer-orthanc-1`: Orthanc DICOM gateway (`http://192.168.0.207:8042`)
* **SQLite Database**:
  * Host path: `/var/lib/docker/volumes/mri-cpu-enhancer_enhancer-data/_data/enhancer.sqlite3`
  * Container path: `/var/lib/enhancer/enhancer.sqlite3`
* **Web UI Credentials**:
  * Admin Portal: `admin` / `adminPassword2026!` ([http://192.168.0.207:8000/admin](http://192.168.0.207:8000/admin))
  * Print Studio: [http://192.168.0.207:8000/admin/print-studio](http://192.168.0.207:8000/admin/print-studio)
  * Orthanc REST / Web: `admin` / `orthancPassword2026!` ([http://192.168.0.207:8042/](http://192.168.0.207:8042/))
* **Viewer URL**:
  * OHIF Viewer: `http://192.168.0.207:8042/ohif/viewer` (configured in `.env` as `OHIF_PUBLIC_URL`)

---

## Direct REST API Endpoints

1. **`POST /api/v1/enhance-dicom`**:
   * Accepts: `file` (multipart DICOM file), `profile` (e.g. `RAD4AI_PRINT_MASTER`), `strength` (`STRONG`/`MEDIUM`), `upscale` (`true`/`false`).
   * Returns: Enhanced `.dcm` binary directly in the response in ~0.15s.

2. **`POST /api/v1/enhance-studio`**:
   * Accepts: DICOM (`.dcm`) or standard image (`.png`, `.jpg`).
   * Parameters: `profile`, `strength`, `upscale`, `send_ge` (`true` to auto-push to `192.168.0.206:4006`).
   * Returns: JSON with base64 before/after previews, execution metrics, download link, and `sent_ge: true/false`.

---

## Critical DICOM Configuration Rules

1. **AE Title Case Sensitivity**:
   * Incoming DICOM C-STORE queries must match `source_aet` case-insensitively (`COLLATE NOCASE` in `store.py`).
   * The GE Advantage Workstation identifies itself as **`adw`** (lowercase).
   * Calling AE Title: `adw`
   * Called AE Title on Orthanc: `MRI03T`
   * `valid_aet` in `admin.py` must preserve case without `.upper()`.

2. **GE Advantage Workstation Ports**:
   * GE AW workstations listen on port **`4006`** (never default port 104).
   * DICOM C-ECHO verification: `AET: adw`, `Host: 192.168.0.206`, `Port: 4006`.

3. **Center 8 (`DarElHa`) Settings**:
   * `enabled = 1`
   * `auto_enhance_enabled = 1`
   * `auto_profile = 'RAD4AI_3T_HD'` (or `'RAD4AI_PRINT_MASTER'`, `'RAD4AI_SPINE_PRO'`, `'RAD4AI_GE_AW'`)
   * `auto_strength = 'STRONG'` (unlocked in backend)
   * `return_policy = 'AUTO_PASS'` (auto-returns when QC passes)
   * `return_ip = '192.168.0.206'`
   * `return_port = 4006`
   * `return_aet = 'adw'`

4. **Profiles & Clinical Safety**:
   * **`RAD4AI_3T_HD`** (RAD4AI 3T-HD True Anatomy): Uses **Kramer-Bruckner Morphological Shock Filter** to compress blurry 0.3T boundaries into 1px razor-sharp cortical/disc edges, plus **3T Trabecular Micro-Texture** to synthesize authentic bone marrow speckle texture, with 100% anatomical fidelity (does not fabricate or alter discs, nerves, or lesions).
   * **`RAD4AI_PRINT_MASTER`**: Optical film-print sigmoid S-curve, pure pitch-black background, and zero-grain soft-tissue denoising.
   * **Loop Prevention**: Intake filters out any series starting with `ENH`, containing `[RAD4AI`, `[ENHANCED]`, or `SYNTH-`.

5. **STRONG Processing Rules**:
   * The restrictions blocking `STRONG` from `auto_enhance_enabled` and `AUTO_PASS` return have been removed from `admin.py`, `intake_worker.py`, and `comparison_worker.py`. `STRONG` now auto-enhances and auto-resends automatically without error.

---

## Clinical Quality & Golden Calibration Rules (Nov 23 Baseline)

1. **GE AW Display Calibration (Brightness & Contrast)**:
   * **Mandatory Setting**: `clinic_brightness = 1.08` (+8%) and `clinic_contrast = 1.18` (+18%) in `system_settings`.
   * **Clinical Reason**: Zeroing them to `1.0/1.0` reduces contrast by 18% and drops $p_{99}$ by >20% (from 666 down to 530), making spine and knee scans look dark, muddy gray, and low-contrast on GE AW diagnostic monitors.

2. **Axial & Sagittal Core Processing**:
   * **Spinal Canal / Foraminal Expansion**: Axial views use `enhanced = np.clip(np.power(enhanced, 0.82) * 1.08, 0.0, 1.0)` to maintain natural high clarity without synthetic artifacts.
   * **Coil Glare Balance**: Do not apply aggressive local coil attenuation (<0.70x) or harsh highlight rolloffs that turn subcutaneous fat and cancellous bone into muddy gray.
   * **2.5D Multi-Slice Coherence**: Preserves anatomical fascial planes and cauda equina nerve roots across adjacent slices while eliminating 0.3T k-space noise.

3. **Autonomous DICOM Return Reliability**:
   * **`AUTO_PASS` Threshold**: `comparison_worker` auto-returns both `PASS` and `WARN` scans (`qc_status in ("PASS", "WARN")`) so minor fractional metric variations never silently hold patient scans.
   * **Live C-ECHO Verification**: `return_worker` tests live connection to `adw:4006` rather than relying on stale database flags.
   * **Self-Healing Sweep**: `_heal_unsent_ready_jobs()` runs every 2 seconds in `return_worker` to automatically pick up and deliver any unsent ready scans.
   * **SQLite Concurrency**: Database runs in `WAL` mode (`PRAGMA journal_mode=WAL`) to allow concurrent multi-worker reads/writes without `database is locked` errors.

4. **Telegram Diagnostic Suite & Quality Polling (`@radainewbot`)**:
   * **Per-Scan Satisfaction Poll**: Each enhanced series notification on Telegram now asks: `❓ هل أنت راضٍ عن جودة هذا الفحص؟` with prominent buttons `[ 👍 نعم (راضٍ تماماً) ]` and `[ 👎 لا (غير راضٍ) ]`.
   * **Instant Tech Feedback Action**: Answering "نعم" records instant approval. Answering "لا" (or "غير راضٍ") **automatically and immediately generates and returns all 5 distinct calibration levels** (`L1-SOFT`, `L2-BALANCED`, `L3-CRISP`, `L4-PRINT`, `L5-AI3T`) to GE AW (`adw:4006`) without requiring manual technician commands, enabling the technician to inspect and select their preferred image style directly on the workstation. Quick adjustment buttons (`يحتاج خشونة`, `يحتاج تنعيم`, `ENH-6`) remain accessible.
   * `/delay` (or "فحص التأخير"): Real-time CPU, RAM, per-slice AI speed, queue depth, and stage-by-stage latency breakdown.
   * `/returns` (or "فحص الإرجاع"): Live `adw:4006` C-ECHO check and status of last 6 received series.
   * `/push` (or "دفع الصور"): 1-click force push of any unsent or stuck scans from the last 12 hours.
   * **Automated Medical Reports**: Automated bilingual Gemini text reports after every scan remain disabled (`enable_gemini_report = 0`).

