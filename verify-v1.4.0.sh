#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
docker compose ps

payload="$(curl --retry 15 --retry-delay 3 --retry-connrefused --fail --silent http://127.0.0.1:8000/health)"
version="$(curl --fail --silent http://127.0.0.1:8000/openapi.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["info"]["version"])')"
python3 - "$payload" "$version" <<'PY'
import json, sys
assert json.loads(sys.argv[1])["status"] in {"ok", "degraded"}
assert sys.argv[2] == "1.4.0", sys.argv[2]
PY

docker compose exec -T enhancer python - <<'PY'
import numpy as np
from app.store import auto_eligible_profile_names, connect, initialize
from app.synthetic import (corrupt_volume, difference_map, make_phantom,
                           reference_metrics, result_decision)

initialize()
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
print("ROI_METRICS=PASS")
print("DIFFERENCE_MAP=PASS")
print("AUTOMATION_SAFETY=PASS")
print("DATABASE_MIGRATION=PASS")
PY

echo "HEALTH=PASS"
echo "VERSION=1.4.0"
echo "RESULT=PASS"
