#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/mri
docker compose config -q
docker compose ps

health="$(curl --retry 15 --retry-delay 3 --retry-connrefused --fail --silent http://127.0.0.1:8000/health)"
version="$(curl --fail --silent http://127.0.0.1:8000/openapi.json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])")"

python3 - "$health" "$version" <<'PY'
import json, sys
health = json.loads(sys.argv[1])
assert health["status"] in {"ok", "degraded"}
assert health["return_worker"]["running"] is True
assert sys.argv[2] in {"1.0.0", "1.0.1"}, sys.argv[2]
print("HEALTH=PASS")
print("VERSION=" + sys.argv[2])
print("RETURN_WORKER=PASS")
PY

docker compose exec -T enhancer python - <<'PY'
from app.store import connect
with connect() as db:
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
required = {"return_jobs", "audit_events", "training_datasets", "model_registry"}
assert required <= tables, required - tables
print("DATABASE_MIGRATION=PASS")
PY

echo "RESULT=PASS"
