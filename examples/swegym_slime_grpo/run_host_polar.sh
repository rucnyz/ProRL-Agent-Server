#!/usr/bin/env bash
# Host-side launcher for the SPLIT topology (see PGC_RLCC_HANDOFF.md).
#
# On B300 we cannot run Slime bare-metal (dep hell), so Slime + SGLang run in
# the slimerl/slime-cu130 container while Polar's rollout + gateway run here on
# the host where apptainer works natively. Container uses --network host so
# 127.0.0.1 is shared between the two.
#
# This script:
#   - renders topology.yaml + polar_config.yaml with our box's ports
#   - starts `polar serve_rollout` and `polar serve_gateway` (host venv)
#
# Ports (this box: 8080/8100/9000 are taken by another user, so 18xxx/19xxx):
#   ROLLOUT_PORT  18080   GATEWAY_PORT 18100   SGLANG_ROUTER_PORT 19000
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
POLAR_BIN="${POLAR_BIN:-${PROJECT_ROOT}/.venv/bin/polar}"

ROLLOUT_PORT="${ROLLOUT_PORT:-18080}"
GATEWAY_PORT="${GATEWAY_PORT:-18100}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-19000}"
# Slime's SGLang router binds to the box's primary IP (get_host_info), not
# 127.0.0.1 — so the gateway must proxy to that IP, not loopback.
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(python3 -c 'import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(("8.8.8.8",80)); print(s.getsockname()[0]); s.close()')}"
SGLANG_ROUTER_BASE_URL="http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}"

# Harness override. The minimal SGLang token-id patch (patch_sglang_min.sh)
# only patches the NON-streaming response path, so for trainable token_ids the
# agent must speak non-streaming. `pi` (require_streaming=false) is bundled in
# our agent-CLI dir; the default qwen_code is streaming and yields zero
# trainable tokens with the minimal patch.
HARNESS="${HARNESS:-pi}"
AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node}"
APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${PROJECT_ROOT}/tmp/swegym_apptainer_images}"
export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-/usr/bin/apptainer}"

RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/swegym_slime_grpo}"
mkdir -p "${RUN_DIR}"
TOPOLOGY_PATH="${RUN_DIR}/topology.yaml"
CUSTOM_CONFIG_PATH="${RUN_DIR}/polar_config.yaml"

"${PYTHON_BIN}" - \
  "${SCRIPT_DIR}/topology.yaml" "${TOPOLOGY_PATH}" "${SGLANG_ROUTER_BASE_URL}" \
  "${SCRIPT_DIR}/polar_config.yaml" "${CUSTOM_CONFIG_PATH}" \
  "${AGENT_CLI_DIR}" "${APPTAINER_IMAGE_DIR}" \
  "${ROLLOUT_PORT}" "${GATEWAY_PORT}" "${HARNESS}" <<'PY'
from pathlib import Path
import sys
import yaml

(topo_in, topo_out, router_url, polar_in, polar_out,
 agent_cli_dir, image_dir, rollout_port, gateway_port, harness) = sys.argv[1:]
rollout_port, gateway_port = int(rollout_port), int(gateway_port)

topo = yaml.safe_load(open(topo_in)) or {}
topo.setdefault("rollout", {})["port"] = rollout_port
topo["rollout"]["public_url"] = f"http://127.0.0.1:{rollout_port}"
topo["rollout"]["host"] = "127.0.0.1"
for node in topo.get("gateway", {}).get("nodes", []):
    node["host"] = "127.0.0.1"
    node["port"] = gateway_port
    node["public_url"] = f"http://127.0.0.1:{gateway_port}"
    node.setdefault("sglang", {})["base_url"] = router_url
Path(topo_out).parent.mkdir(parents=True, exist_ok=True)
yaml.safe_dump(topo, open(topo_out, "w"), sort_keys=False)

pc = yaml.safe_load(open(polar_in)) or {}
pc["polar_rollout_url"] = f"http://127.0.0.1:{rollout_port}"
pc["polar_gateway_url"] = f"http://127.0.0.1:{gateway_port}"
pc["polar_agent_cli_dir"] = agent_cli_dir
pc["polar_apptainer_image_dir"] = image_dir
pc.setdefault("polar_task_template", {}).setdefault("agent", {})["harness"] = harness
yaml.safe_dump(pc, open(polar_out, "w"), sort_keys=False)
print(f"rendered {topo_out} (rollout :{rollout_port}, gateway :{gateway_port}, sglang {router_url})")
print(f"rendered {polar_out}")
PY

echo "=== Starting Polar rollout (:${ROLLOUT_PORT}) + gateway (:${GATEWAY_PORT}) on host ==="
"${POLAR_BIN}" serve_rollout -c "${TOPOLOGY_PATH}" &
ROLLOUT_PID=$!
sleep 3
"${POLAR_BIN}" serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 &
GATEWAY_PID=$!
sleep 3

cleanup() { kill "${ROLLOUT_PID}" "${GATEWAY_PID}" 2>/dev/null || true; }
trap cleanup EXIT

curl -sf "http://127.0.0.1:${ROLLOUT_PORT}/health" && echo " <- rollout healthy" || {
  echo "Polar rollout not healthy"; exit 1; }

echo
echo "Polar is up. Rendered configs:"
echo "  topology:     ${TOPOLOGY_PATH}"
echo "  polar_config: ${CUSTOM_CONFIG_PATH}"
echo "Now launch the container side: bash examples/swegym_slime_grpo/run_container_slime.sh"
echo "(Ctrl-C here to stop Polar after training ends.)"
wait
