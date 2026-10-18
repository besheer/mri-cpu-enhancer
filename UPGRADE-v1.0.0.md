# Upgrade to v1.0.0

Research-only release. This package is not a medical device and is not approved for diagnosis.

## Highlights

- Study cards with thumbnails, expand/collapse, and retained UI state.
- Arabic/English direction switch and persistent light/dark theme.
- Batch selection and AJAX queue actions without jumping to the page top.
- Center network mode: VPN, public static IP, NAT forward, Agent Pull, or no return.
- Return policy: disabled, manual, or automatic only after Quality Gate PASS.
- Independent return queue with C-ECHO prerequisite, retries, and audit events.
- MONAI dataset registration workspace. Training and model activation remain disabled until a suitable dataset and validation protocol exist.

## Safe upgrade

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v1.0.0.tar.gz
mkdir -p datasets
docker compose config -q
docker compose build --no-cache enhancer
docker compose up -d --force-recreate enhancer
curl --retry 15 --retry-delay 3 --retry-connrefused --fail http://127.0.0.1:8000/health
```

Existing SQLite data is migrated in place. Keep the existing `.env`; the archive contains only `.env.example`.

## Return activation

1. Edit a center and select its network mode.
2. Keep Return Policy on Manual during commissioning.
3. Run Test C-ECHO from the dashboard.
4. Process and review a derived series.
5. Use Approve & Return. Enable Auto if PASS only after end-to-end acceptance testing.

Never expose Orthanc HTTP (8042) or the enhancer API (8000) directly to the WAN.
