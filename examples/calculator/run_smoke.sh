#!/usr/bin/env bash
# 1-gateway smoke test launcher for our fork.
# Assumes SGLang container is already running and reachable at the URL in
# topology-smoke.yaml. Submit script reads ./topology.yaml, so we swap in
# the smoke topology before submitting and restore upstream after.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TOPO="${HERE}/topology.yaml"
SMOKE="${HERE}/topology-smoke.yaml"

if [[ ! -f "${SMOKE}" ]]; then
  echo "missing ${SMOKE}" >&2
  exit 1
fi

# Swap topologies for the duration of the submit, then restore.
trap 'mv -f "${TOPO}.bak" "${TOPO}" 2>/dev/null || true' EXIT
mv "${TOPO}" "${TOPO}.bak"
cp "${SMOKE}" "${TOPO}"

# Use our pinned local claude-code if present; otherwise fall through to system.
LOCAL_CLAUDE="${HERE}/../../.local-npm/node_modules/.bin"
if [[ -d "${LOCAL_CLAUDE}" ]]; then
  export PATH="${LOCAL_CLAUDE}:${PATH}"
fi

PROJ_ROOT="${HERE}/../.."
PY="${PROJ_ROOT}/.venv/bin/python"
"${PY}" "${HERE}/submit_calculator_task.py" "$@"
