#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"

cleanup() {
  docker compose --profile e2e stop e2e-pacs >/dev/null 2>&1 || true
  docker compose --profile e2e rm -f e2e-pacs >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup
docker compose --profile e2e build e2e-pacs
docker compose --profile e2e up -d --no-build e2e-pacs

set +e
docker compose exec -T enhancer python -m app.e2e_test
rc=$?
set -e
echo "E2E_SCRIPT_RC=$rc"
exit "$rc"
