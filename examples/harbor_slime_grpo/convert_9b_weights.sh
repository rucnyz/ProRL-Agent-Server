#!/usr/bin/env bash
# Convert Qwen3.5-9B HF weights → Megatron torch_dist for Slime training.
#
# Qwen3.5-9B is the SAME hybrid-attention architecture as the 4B
# (Qwen3_5ForConditionalGeneration; 24 GatedDeltaNet-linear + 8 full attention
# layers, head_dim 256, vocab 248320, rope 1e7). It differs from the 4B ONLY in
# hidden_size (2560→4096) and ffn (9216→12288). SkyRL trained 9B on the Harbor
# tasks, so 9B (unlike the too-weak 4B) actually solves some of them → non-zero
# reward variance for GRPO.
#
# Run inside slimerl/slime-cu130:local (which already has slime + Megatron):
#   docker run --rm --gpus '"device=0"' --shm-size 32g --ipc=host \
#     -v <repo>:/workspace/ProRL-Agent-Server \
#     -v ~/.cache/huggingface:/root/.cache/huggingface -e HF_TOKEN=$HF_TOKEN \
#     slimerl/slime-cu130:local bash -lc 'bash examples/harbor_slime_grpo/convert_9b_weights.sh'
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ProRL-Agent-Server}"
SLIME_DIR="${SLIME_DIR:-/root/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-9B}"
OUTPUT_DIR="${TORCH_DIST_DIR:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-9B_torch_dist}"
mkdir -p "$OUTPUT_DIR"

CUDA_DEVICE_MAX_CONNECTIONS=1 \
PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src" \
torchrun --nproc_per_node 1 \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec" \
    --disable-bias-linear --qk-layernorm --group-query-attention \
    --num-attention-heads 16 --num-query-groups 4 --kv-channels 256 \
    --num-layers 32 --hidden-size 4096 --ffn-hidden-size 12288 \
    --use-gated-attention --normalization RMSNorm --apply-layernorm-1p \
    --position-embedding-type rope --norm-epsilon 1e-6 --rotary-percent 0.25 \
    --swiglu --vocab-size 248320 --rotary-base 10000000 --attention-output-gate \
    --hf-checkpoint "$HF_CHECKPOINT" --save "$OUTPUT_DIR" \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --no-gradient-accumulation-fusion
echo "Done: ${OUTPUT_DIR}"
