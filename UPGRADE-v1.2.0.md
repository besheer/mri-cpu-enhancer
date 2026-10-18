# Upgrade to v1.2.0

## Included

- Separate Receive, Enhancement and Return queue management page.
- Safe archive/clear of completed records without deleting DICOM images.
- Cancel queued enhancement jobs and retry failed return jobs.
- Pause/Resume controls for Intake, Enhancement and Return workers.
- Maintenance mode that pauses all three workers.
- Persistent audit events for queue and service-control operations.
- Synthetic end-to-end DICOM test with a temporary mock PACS.

## Upgrade

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v1.2.0.tar.gz
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
./verify-v1.2.0.sh
```

## End-to-end test

The test uses only synthetic DICOM data and creates a temporary test center.

```bash
cd /home/mri
./run-e2e.sh
```

Success ends with `E2E_RESULT=PASS` and `E2E_SCRIPT_RC=0`.
