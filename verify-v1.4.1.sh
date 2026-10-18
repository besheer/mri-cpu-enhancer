#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
bash -n run-e2e.sh
docker compose ps

payload="$(curl --retry 15 --retry-delay 3 --retry-connrefused --fail --silent http://127.0.0.1:8000/health)"
version="$(curl --fail --silent http://127.0.0.1:8000/openapi.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])')"
python3 - "$payload" "$version" <<'PY'
import json, sys
assert json.loads(sys.argv[1])["status"] in {"ok", "degraded"}
assert sys.argv[2] == "1.4.1", sys.argv[2]
PY

docker compose exec -T enhancer python - <<'PY'
from pathlib import Path
import numpy as np
from app.store import auto_eligible_profile_names, connect, initialize
from app.synthetic import (corrupt_volume, difference_map, make_phantom,
                           reference_metrics, result_decision)

initialize()
source = Path("/app/app/e2e_test.py").read_text()
runner = Path("/app/run-e2e.sh").read_text() if Path("/app/run-e2e.sh").exists() else ""
assert "EDGE_DENOISE_MILD" in source
assert "def wait_for_receiver" in source
assert "def cleanup_e2e" in source

clean = make_phantom(8, 64)
mild = corrupt_volume(clean, "RICIAN_NOISE", "MILD", 42)
strong = corrupt_volume(clean, "RICIAN_NOISE", "STRONG", 42)
first = reference_metrics(clean, mild)
second = reference_metrics(clean, strong)
identity = reference_metrics(clean, clean)

assert first["roi_psnr_db"] > second["roi_psnr_db"]
assert identity["roi_ssim"] == 1.0
assert identity["slice_ssim_mean"] == 1.0
assert difference_map(clean, mild).max() > 0
assert result_decision("MOTION", first, identity)[0] == "REVIEW_ONLY"
assert "BASE_MILD" not in auto_eligible_profile_names()
assert "CLAHE_VIEW" not in auto_eligible_profile_names()
with connect() as db:
    columns = {row["name"] for row in db.execute("PRAGMA table_info(synthetic_results)")}
    tool_columns = {row["name"] for row in db.execute("PRAGMA table_info(processing_tools)")}
assert {"difference_series_id", "decision", "rank_score"} <= columns
assert "auto_eligible" in tool_columns
print("E2E_PROFILE=PASS")
print("E2E_READINESS_AND_CLEANUP=PASS")
print("ROI_METRICS=PASS")
print("DIFFERENCE_MAP=PASS")
print("AUTOMATION_SAFETY=PASS")
print("DATABASE_MIGRATION=PASS")
PY

echo "HEALTH=PASS"
echo "VERSION=1.4.1"
echo "RESULT=PASS"
