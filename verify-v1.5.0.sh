#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"
bash -n run-e2e.sh
docker compose config -q
docker compose ps

health="$(curl --retry 15 --retry-delay 3 --retry-connrefused --fail --silent http://127.0.0.1:8000/health)"
openapi="$(curl --fail --silent http://127.0.0.1:8000/openapi.json)"
python3 - "$health" "$openapi" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
api = json.loads(sys.argv[2])
assert health["status"] in {"ok", "degraded"}
assert api["info"]["version"] == "1.5.0", api["info"]["version"]
paths = api["paths"]
assert "/admin/protocols" in paths
assert "/admin/protocols/{protocol_id}/review" in paths
assert "/admin/protocols/{protocol_id}/enhance" in paths
assert "/admin/protocols/export.csv" in paths
print("PROTOCOL_ROUTES=PASS")
PY

docker compose exec -T enhancer python - <<'PY'
from app.protocol_study import calculate_reduction
from app.store import connect, initialize
from pathlib import Path

initialize()
assert calculate_reduction(240, 180) == 25.0
with connect() as db:
    tables = {row["name"] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    columns = {row["name"] for row in db.execute("PRAGMA table_info(comparison_jobs)")}
    trigger = db.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name='block_protocol_study_return'").fetchone()
assert {"protocol_studies", "protocol_reviews"} <= tables
assert "protocol_study_id" in columns
assert trigger and "Protocol Study results cannot be returned" in trigger["sql"]
worker_source = Path("/app/app/comparison_worker.py").read_text()
assert "protocol_protected" in worker_source
print("PROTOCOL_DATABASE=PASS")
print("PROTOCOL_RETURN_GUARD=PASS")
print("PROTOCOL_WORKER_GUARD=PASS")
print("SCAN_TIME_CALCULATION=PASS")
PY

echo "HEALTH=PASS"
echo "VERSION=1.5.0"
echo "RESULT=PASS"
