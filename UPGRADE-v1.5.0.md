# Upgrade to v1.5.0

Version 1.5.0 adds a research-only Protocol Acceleration Study workflow while
preserving the v1.4.1 DICOM receive, enhancement, comparison, and return flow.

## New workflow

- Pair original Standard and Fast series from the same Orthanc study/patient.
- Record anatomy, sequence, scanner, coil, protocol changes, and scan times.
- Calculate scan-time reduction automatically.
- Enhance only the Fast input with a selected comparison profile.
- Compare Standard, Fast, and derived series in OHIF.
- Store append-only medical reviews with five rating dimensions.
- Record ACCEPT, REVIEW, or REJECT decisions and export CSV.
- Block Protocol Study images from Return Queue at the database level.

## Upgrade

Keep the existing `.env` and Docker volumes:

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v1.5.0.tar.gz
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
./verify-v1.5.0.sh
```

Open:

```text
https://mri.radbridges.com/admin/protocols
```

Create the Protocol Study before approving or returning any experimental
series. Use original non-ENHANC series and keep branch Auto Enhancement off
during protocol acquisition experiments.

> Research only. This release does not validate a scan-time reduction for
> clinical use and does not authorize protocol changes without local medical,
> MR safety, and regulatory review.
