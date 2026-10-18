# Upgrade to v1.3.0

This release adds the Synthetic MRI Lab at `/admin/synthetic`.

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v1.3.0.tar.gz
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
./verify-v1.3.0.sh
```

Recommended first run: `MIXED_03T`, artifact severity `MEDIUM`, enhancement level
`MEDIUM`, with Baseline, Denoise, Edge Denoise, and N4 Edge Denoise selected.

Synthetic studies never enter the automatic return queue.
