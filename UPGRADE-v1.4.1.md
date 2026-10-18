# Upgrade to v1.4.1

This maintenance release makes the end-to-end DICOM test repeatable and
self-cleaning. It preserves the v1.4.0 processing, dashboard, database, and
safety behavior.

## Changes

- Uses `EDGE_DENOISE_MILD/MILD`, an automation-eligible conservative profile.
- Waits for Compose DNS, TCP port 11112, and DICOM C-ECHO before sending.
- Records the E2E PASS/FAIL result in the audit log.
- Archives temporary test queue rows and removes the temporary test center.
- Stops and removes the mock PACS container after either PASS or FAIL.

## Upgrade

Keep the existing `.env` and Docker volumes. From `/home/mri`:

```bash
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v1.4.1.tar.gz
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
./verify-v1.4.1.sh
```

Run the complete DICOM send, enhance, and return test:

```bash
./run-e2e.sh
```

Expected final lines include `E2E_PACS_READY=PASS`, `PIPELINE_STATUS=SENT`,
`RETURNED_INSTANCES=3`, `E2E_RESULT=PASS`, and `E2E_SCRIPT_RC=0`.

> Research only; not approved for diagnostic use.
