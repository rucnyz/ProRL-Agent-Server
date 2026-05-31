#!/usr/bin/env bash
# Container-side launcher for the SPLIT topology (see PGC_RLCC_HANDOFF.md).
# Runs INSIDE slimerl/slime-cu130:local with --network host and 4 GPUs.
# Polar rollout+gateway must already be up on the host (run_host_polar.sh).
#
# Train args mirror examples/swegym_slime_grpo/run.sh, scaled to a 4-GPU box:
#   GPU 0-1  Megatron GRPO train (TP=2)
#   GPU 2-3  SGLang inference (2 engines, TP=1), slime-managed
# 8x layout in run.sh is the reference; keep them in sync when changing args.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/ProRL-Agent-Server}"
SCRIPT_DIR="${PROJECT_ROOT}/examples/swegym_slime_grpo"
SLIME_DIR="${SLIME_DIR:-/root/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-/root/Megatron-LM}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/swegym_slime_grpo}"

HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_ROOT}/tmp/ckpt/swegym_slime_grpo_qwen35_4b}"
mkdir -p "${SAVE_DIR}" "${PROJECT_ROOT}/logs"

CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/polar_config.yaml}"
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/swegym_train_293.jsonl}"
if [ ! -f "${CUSTOM_CONFIG_PATH}" ]; then
    echo "ERROR: ${CUSTOM_CONFIG_PATH} missing — run run_host_polar.sh on the host first." >&2
    exit 1
fi

# 4-GPU split
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-4}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6380}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8266}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-60000}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-19000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"

if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="${SAVE_DIR}"
else
    LOAD_DIR="${REF_LOAD}"
fi

MODEL_ARGS=(
    --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
    --disable-bias-linear --qk-layernorm --group-query-attention
    --num-attention-heads 16 --num-query-groups 4 --kv-channels 256
    --num-layers 32 --hidden-size 2560 --ffn-hidden-size 9216
    --use-gated-attention --normalization RMSNorm --apply-layernorm-1p
    --position-embedding-type rope --norm-epsilon 1e-6 --rotary-percent 0.25
    --swiglu --vocab-size 248320 --rotary-base 10000000 --attention-output-gate
)

CUDNN_LIB="/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib"
CU13_LIB="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"
NVRTC12_LIB="/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib"
RUNTIME_LD="${CU13_LIB}:${NVRTC12_LIB}:${CUDNN_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WANDB_DIR\": \"${PROJECT_ROOT}/logs\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\"
  }
}"

# Apply the SGLang token-id emission patch (idempotent). Without it the
# gateway-recorded trajectories carry no token_ids and slime_bridge drops
# every trace as "zero trainable tokens". patch_sglang_min.sh patches only the
# non-streaming response path → pair with a non-streaming harness (pi).
PATCH_SGLANG="${PATCH_SGLANG:-1}"
if [ "${PATCH_SGLANG}" = "1" ]; then
    echo "=== Applying SGLang token-id patch (patch_sglang_min.sh) ==="
    bash "${PROJECT_ROOT}/scripts/patch/patch_sglang_min.sh" || {
        echo "WARN: patch_sglang_min.sh did not apply cleanly; continuing" >&2; }
fi

echo "=== Starting Ray (${RAY_NUM_GPUS} GPUs) ==="
ray stop --force 2>/dev/null || true
sleep 1
ray start --head --node-ip-address 127.0.0.1 --port "${RAY_GCS_PORT}" --dashboard-port "${RAY_DASHBOARD_PORT}" --num-gpus "${RAY_NUM_GPUS}" --disable-usage-stats

echo "=== Launching train_async.py (split mode; Polar on host) ==="
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${REF_LOAD}" \
    --load "${LOAD_DIR}" \
    --save "${SAVE_DIR}" \
    --save-interval "${SAVE_INTERVAL}" \
    --update-weights-interval 1 \
    --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
    --custom-rm-path slime_bridge.reward.reward_func \
    --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards \
    --custom-config-path "${CUSTOM_CONFIG_PATH}" \
    --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer \
    --prompt-data "${PROMPT_DATA}" \
    --input-key prompt --label-key label --metadata-key metadata \
    --rollout-shuffle --reward-key score --num-epoch 1 \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
    --rollout-max-response-len 16000 --rollout-max-prompt-len 32000 \
    --dynamic-history --num-steps-per-rollout 1 \
    --tensor-model-parallel-size 2 --sequence-parallel \
    --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --log-probs-chunk-size 256 \
    --advantage-estimator grpo --normalize-advantages --use-tis \
    --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl \
    --entropy-coef 0.0 --eps-clip 0.2 --eps-clip-high 0.28 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
    --adam-beta1 0.9 --adam-beta2 0.98 \
    --attention-dropout 0.0 --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 \
    --attention-backend auto --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static 0.8 \
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}" \
    --sglang-tool-call-parser qwen3_coder \
    --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}" \
    --sglang-router-port "${SGLANG_ROUTER_PORT}" \
    "$@"
