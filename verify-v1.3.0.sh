#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
docker compose ps

payload="$(curl --retry 15 --retry-delay 3 --retry-connrefused --fail --silent http://127.0.0.1:8000/health)"
version="$(curl --fail --silent http://127.0.0.1:8000/openapi.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])')"
python3 - "$payload" "$version" <<'PY'
import json, sys
assert json.loads(sys.argv[1])["status"] in {"ok", "degraded"}
assert sys.argv[2] == "1.3.0", sys.argv[2]
PY

docker compose exec -T enhancer python - <<'PY'
import numpy as np
from app.store import connect, initialize
from app.synthetic import corrupt_volume, make_phantom, reference_metrics

initialize()
clean = make_phantom(8, 64)
mild = corrupt_volume(clean, "RICIAN_NOISE", "MILD", 42)
strong = corrupt_volume(clean, "RICIAN_NOISE", "STRONG", 42)
mixed_a = corrupt_volume(clean, "MIXED_03T", "MEDIUM", 7)
mixed_b = corrupt_volume(clean, "MIXED_03T", "MEDIUM", 7)

assert clean.shape == (8, 64, 64)
assert clean.dtype == np.uint16
assert np.array_equal(mixed_a, mixed_b)
assert not np.array_equal(clean, mixed_a)
assert reference_metrics(clean, mild)["psnr_db"] > reference_metrics(clean, strong)["psnr_db"]
assert reference_metrics(clean, clean)["global_ssim"] == 1.0
with connect() as db:
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert {"synthetic_runs", "synthetic_results"} <= tables
print("SYNTHETIC_ENGINE=PASS")
print("DATABASE_MIGRATION=PASS")
PY

echo "HEALTH=PASS"
echo "VERSION=1.3.0"
echo "RESULT=PASS"
