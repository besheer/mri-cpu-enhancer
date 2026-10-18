# Upgrade to v1.2.1

E2E test correction only. Production Quality Gate and Auto Return rules are unchanged.

The synthetic baseline can produce a legitimate QC WARN. The updated test verifies that WARN blocks automatic return, then explicitly queues a synthetic manual return to test C-ECHO and C-STORE transport without weakening production safety thresholds.

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v1.2.1.tar.gz
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
./verify-v1.2.1.sh
./run-e2e.sh
```
