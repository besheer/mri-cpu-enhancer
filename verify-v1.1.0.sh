#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/mri
docker compose config -q
docker compose ps
health="$(curl --retry 15 --retry-delay 3 --retry-connrefused --fail --silent http://127.0.0.1:8000/health)"
version="$(curl --fail --silent http://127.0.0.1:8000/openapi.json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])")"
python3 - "$health" "$version" <<'PY'
import json, sys
h=json.loads(sys.argv[1])
assert sys.argv[2] == "1.1.0", sys.argv[2]
assert h["intake_worker"]["running"] is True
assert h["return_worker"]["running"] is True
assert h["system"]["orthanc"] == "UP"
print("HEALTH=PASS")
print("VERSION=1.1.0")
print("INTAKE_WORKER=PASS")
print("RETURN_WORKER=PASS")
print("ORTHANC=PASS")
PY
docker compose exec -T enhancer python - <<'PY'
from app.store import connect
with connect() as db:
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert {"inbound_series","comparison_jobs","return_jobs","audit_events"} <= tables
print("DATABASE_MIGRATION=PASS")
PY
echo "RESULT=PASS"
