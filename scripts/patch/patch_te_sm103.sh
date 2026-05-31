#!/usr/bin/env bash
# Patch TransformerEngine 2.10 to allow FlashAttention for head_dim>192 on B300
# (sm103 / Blackwell Ultra).
#
# TE 2.10's attention backend selector hard-codes the set of compute capabilities
# that support flash-attention with head_dim_qk in (192, 256]:
#
#   transformer_engine/pytorch/attention/dot_product_attention/utils.py
#     head_dim_qk > 192
#     and device_compute_capability not in ((8, 0), (9, 0), (10, 0), (12, 0))
#
# i.e. sm80 (A100), sm90 (H100), sm100 (B200), sm120 (RTX50) — but NOT sm103
# (B300). Qwen3.5 uses kv_channels=256 (head_dim=256), so on B300 TE disables
# flash-attn; cuDNN 9.16 fused-attn also lacks head_dim=256 support, and the
# unfused backend is disabled for the THD (packed, variable-length) layout slime
# uses for GRPO. Result: "No dot product attention backend is available".
#
# sm103 is the same Blackwell family as sm100, and the installed flash-attn
# 2.7.4 kernels run correctly on it (verified: THD + head_dim=256 forward on a
# real B300). So we just add (10, 3) to TE's whitelist. One-line, idempotent,
# no rebuild — keeps THD packing and full flash-attn performance.
set -euo pipefail

PATCH_PYTHON="${PATCH_PYTHON:-python3}"
TE_UTILS="$("${PATCH_PYTHON}" - <<'PY'
import os, transformer_engine.pytorch.attention.dot_product_attention.utils as u
print(os.path.abspath(u.__file__))
PY
)"

if [ ! -f "${TE_UTILS}" ]; then
    echo "ERROR: could not locate TE utils.py (${TE_UTILS})" >&2
    exit 1
fi

OLD='((8, 0), (9, 0), (10, 0), (12, 0))'
NEW='((8, 0), (9, 0), (10, 0), (10, 3), (12, 0))'

if grep -qF "${NEW}" "${TE_UTILS}"; then
    echo "patch_te_sm103: already patched (${TE_UTILS})"
    exit 0
fi

if ! grep -qF "${OLD}" "${TE_UTILS}"; then
    echo "WARN: expected whitelist '${OLD}' not found in ${TE_UTILS}; TE version may differ — skipping" >&2
    exit 0
fi

sed -i "s/${OLD//\//\\/}/${NEW//\//\\/}/g" "${TE_UTILS}"
echo "patch_te_sm103: added (10, 3) to TE flash-attn head_dim>192 whitelist in ${TE_UTILS}"
grep -n "(10, 3)" "${TE_UTILS}" | head
