#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/mri
docker compose config -q
health="$(curl --retry 15 --retry-delay 3 --retry-connrefused --fail --silent http://127.0.0.1:8000/health)"
version="$(curl --fail --silent http://127.0.0.1:8000/openapi.json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['version'])")"
python3 - "$health" "$version" <<'PY'
import json,sys
h=json.loads(sys.argv[1])
assert sys.argv[2] == "1.2.1", sys.argv[2]
assert h["intake_worker"]["running"] is True
assert h["return_worker"]["running"] is True
assert h["system"]["orthanc"] == "UP"
print("HEALTH=PASS")
print("VERSION=1.2.1")
print("WORKERS=PASS")
print("ORTHANC=PASS")
PY
echo "RESULT=PASS"
