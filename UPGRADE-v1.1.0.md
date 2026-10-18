# Upgrade to v1.1.0

## Added

- Live CPU, RAM, disk and enhancer-process memory dashboard.
- 5-second CPU/RAM chart and Orthanc reachability status.
- Receive, enhancement and return pipeline counters.
- Per-job slice progress.
- Persistent inbound-series queue.
- Per-branch automatic enhancement profile and strength.
- Automatic return only when the branch policy is AUTO_PASS, Quality Gate is PASS, and C-ECHO previously passed.

## Safety

New centers and existing centers default to manual enhancement and manual return. Enable automation only after C-ECHO and an end-to-end test.

## Upgrade

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v1.1.0.tar.gz
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
./verify-v1.1.0.sh
```

After upgrading, use `Ctrl+F5` in the browser.
