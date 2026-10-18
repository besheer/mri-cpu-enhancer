# Upgrade to v1.0.1

Dark-mode contrast hotfix for text, secondary labels, links, form fields,
placeholders, table headers, buttons, alerts, and status badges.

No database or DICOM workflow changes from v1.0.0.

```bash
cd /home/mri
docker compose stop enhancer
tar -xzf MRI-03T-CPU-Enhancer-v1.0.1.tar.gz
docker compose config -q
docker compose build enhancer
docker compose up -d --force-recreate enhancer
```

Then force-refresh the browser with `Ctrl+F5` so the embedded CSS is reloaded.
