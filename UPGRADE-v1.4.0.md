# Upgrade to v1.4.0

This release strengthens benchmark measurements and automation safety.

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v1.4.0.tar.gz
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
./verify-v1.4.0.sh
```

Important migration behavior:

- Centers configured for automatic `BASE_MILD`, `CLAHE_VIEW`, or `STRONG` are safely
  switched to manual mode. No center or route is deleted.
- Baseline remains available in the Synthetic Lab for comparison.
- Auto Return still requires an automation-eligible profile, a non-STRONG level,
  successful C-ECHO, and QC PASS.
- Synthetic studies and difference maps never enter intake or return queues.

Start with a `MIXED_03T / MEDIUM` run. Use the all-level benchmark only when the
server is otherwise idle because it can generate hundreds of synthetic DICOM instances.
