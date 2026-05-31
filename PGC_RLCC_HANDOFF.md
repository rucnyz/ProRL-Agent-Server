# PGC-RLCC Handoff（rucnyz fork, branch `pgc-rlcc`）

「在 Polar (NVIDIA NeMo) + Slime 上用 Claude Code / Codex 等真实 agent harness 跑 RL」的实验记录。本文档是给自己 / 同学复现用的，**会过时**；以本仓库实际代码为准。

最后更新时间：see `git log -1` on this branch。

## 0. 一句话现状

- ✅ **Polar 端到端 rollout smoke test 在 B300 上跑通**（reward 1.0，calculator example, claude_code harness, SGLang docker image）。
- 🚧 **Slime 训练循环在 B300 + cu130 bare-metal pip 装行不通**（numpy/scipy/transformers 与 Slime 强制 numpy<2 死锁）。
- 🚧 **正在尝试**用 Slime 官方 docker 构建 `ENABLE_CUDA_13=1` 镜像（专为 GB300 设计）来绕过 bare-metal 死锁。

## 1. 工作目录与仓库

| 路径 | 内容 |
|---|---|
| `/scratch/yuzhou/projects/ProRL-Agent-Server/` | 本仓库（rucnyz/ProRL-Agent-Server fork） |
| `/scratch/yuzhou/projects/ProRL-Agent-Server/slime/` | 同级 clone Slime `v0.2.4`（独立 .git） |
| `/scratch/yuzhou/projects/ProRL-Agent-Server/Megatron-LM/` | 同级 clone Megatron-LM @ commit `3714d81`（独立 .git） |
| `/scratch/yuzhou/projects/ProRL-Agent-Server/.venv/` | Polar / 实验用 Python venv（Python 3.13） |
| `/scratch/yuzhou/projects/ProRL-Agent-Server/.local-npm/` | 本地 pin `@anthropic-ai/claude-code@2.1.116`（不污染全局 claude） |
| `/scratch/yuzhou/projects/ProRL-Agent-Server/tmp/` | swegym apptainer images, weight ckpts 等（已 .gitignore） |

Git 远端：
- `origin` → `https://github.com/rucnyz/ProRL-Agent-Server.git`（我们的 fork）
- `upstream` → `https://github.com/NVIDIA-NeMo/ProRL-Agent-Server.git`

工作分支：`pgc-rlcc`

## 2. 已验证可用：Polar Rollout Smoke Test

完整流程：
```
submit_calculator_task.py claude_code
   → polar rollout (18080) → polar gateway (18100)
   → docker run polar-localhost-calculator:latest（容器内 npm install claude-code 2.1.111）
   → claude (设 ANTHROPIC_BASE_URL → polar gateway proxy)
   → polar gateway → SGLang docker (18000, Qwen3.5-4B)
   → Claude Code 写 calculator.py → evaluator 跑 5/5 tests
   → reward = 1.0
```

### 复现命令（B300 + cu130）

1. 装 polar：
   ```bash
   cd /scratch/yuzhou/projects/ProRL-Agent-Server
   uv venv
   uv pip install -e .
   ```
2. 拉官方 SGLang cu130 docker image（30.1GB）：
   ```bash
   docker pull lmsysorg/sglang:latest-cu130-runtime
   ```
3. 装本地 `claude` CLI（不动全局）：
   ```bash
   mkdir -p .local-npm && cd .local-npm
   npm init -y >/dev/null && npm install @anthropic-ai/claude-code@2.1.116
   ```
4. Build polar runtime image：
   ```bash
   .venv/bin/python examples/calculator/build_image.py
   ```
5. 起 SGLang 容器（注意我们这台机 8000/8080/8100/9000 都被占，所以端口都改 18xxx，**别人的机器请改回标准端口**）：
   ```bash
   docker run -d --name polar-sglang-0 \
     --gpus '"device=0"' --shm-size 32g \
     -p 18000:30000 \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     --env HF_TOKEN=$HF_TOKEN \
     --env TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
     --ipc=host \
     --entrypoint /bin/bash \
     lmsysorg/sglang:latest-cu130-runtime \
     -lc "pip install distro && python3 -m sglang.launch_server \
       --model-path Qwen/Qwen3.5-4B --host 0.0.0.0 --port 30000 \
       --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
       --mem-fraction-static 0.7 --context-length 262144 --trust-remote-code"
   ```
   等容器日志出现 `The server is fired up and ready to roll!`。
6. 起 Polar：
   ```bash
   .venv/bin/polar serve_rollout -c examples/calculator/topology-smoke.yaml &
   .venv/bin/polar serve_gateway -c examples/calculator/topology-smoke.yaml --node-id localhost-node-01 &
   .venv/bin/polar status -c examples/calculator/topology-smoke.yaml   # 应该全绿
   ```
7. 跑 smoke：
   ```bash
   bash examples/calculator/run_smoke.sh claude_code
   # 期望: Reward 1.0
   ```

### 已知 issue / workarounds（针对 lmsysorg cu130 image）

| 现象 | 原因 | workaround |
|---|---|---|
| `ModuleNotFoundError: No module named 'distro'` 启动失败 | 官方 image 漏装 distro（openai SDK 2.6.1 Linux 硬依赖） | entrypoint 改 `bash -lc "pip install distro && python3 -m sglang.launch_server ..."` |
| `ptxas fatal: Value 'sm_103a' is not defined` | Triton 自带 ptxas 老 | 设 `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` |
| torchcodec 一串 traceback | 容器里没 ffmpeg；torchcodec 非核心模块 | 忽略（与 SGLang 推理无关） |

### 端口对照（我们这台机的占用情况）

| 用途 | 默认 | 本机用 | 说明 |
|---|---|---|---|
| SGLang | 30000 (内) | 18000 (host) | 8000 被占 |
| Polar rollout | 8080 | 18080 | 8080 被占 |
| Polar gateway | 8100 | 18100 | OK |
| SGLang router (Slime) | 9000 | 待定 | 9000 被占 |
| Ray dashboard | 8265 | 待定 | OK |

如果你换机器，把 `examples/calculator/topology-smoke.yaml` 改回 8000/8080/8100。

## 3. 进行中：Slime 训练循环（B300 + cu130）

### 已完成的准备

- `git clone --branch v0.2.4 https://github.com/THUDM/slime.git slime`（已 patch `slime/docker/patch/latest/megatron.patch` + Polar `scripts/patch/patch_slime.sh`）
- `git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM`，checkout `3714d81`，apply `slime/docker/patch/latest/megatron.patch`（加 `--use-gated-attention` 等 arg）
- `uv pip install -e slime Megatron-LM mbridge`
- `prepare_data.py` → `swegym_train_293.jsonl` 293 个 instance
- `prepare_apptainer_images.py --instance-id getmoto__moto-{7365,4950,6178}` → 3 个 SIF (1.1GB each) + node + 5 agent CLI（codex, claude-code, qwen-code, opencode, pi）@ `tmp/swegym_agent_cli/opt_node/`

### bare-metal 卡点（**不要走这条路**）

走 bare-metal `uv pip install` 在 B300 + cu130 + Python 3.13 上撞到下面 deadlock：

```
Slime → assert numpy 1.x（Megatron 兼容性）
       → scipy 1.18（latest）需要 numpy 2.x → 冲突
       → scipy 1.13.1（最后兼容 numpy 1.x）无 cp313 wheel，要 OpenBLAS 源码 build
       → transformers 5.6.0 内部 scipy import 失败 → PreTrainedModel 不可 import
```

试过：

| 步骤 | 结果 |
|---|---|
| sglang 0.5.10 + sgl-kernel 0.3.21 | sgl-kernel 0.3.21 build against torch 2.9 ABI（`c10_cuda_check_implementation(int, ...)`）。我们 torch 2.10+/cu130 ABI 是 `(unsigned int, ...)` → undefined symbol |
| torch 2.10+cu130 + sglang 0.5.12.post1 + sglang-kernel 0.4.2.post2+cu130 | sglang 可 import，但 transformers 被 sglang downgrade 到 5.6 → `kernels` 0.15.1 `LayerRepository` 要 revision/version → kernels 0.11.0 修了 |
| 然后 numpy 1.26 + scipy 1.18 | scipy 用 `np.long`（numpy 1.x 没有） → `from transformers import PreTrainedModel` 炸 |

**结论：bare-metal 是死胡同。** Slime 官方设计就是 docker。

### docker 路线（当前路径）

Slime `docker/Dockerfile` 提供 `ENABLE_CUDA_13=1` 专为 GB300/B300 设计：

```dockerfile
ARG ENABLE_CUDA_13=0
...
# Triton patched for B300/sm_103a
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    git clone -b feat/v350_plus_8045 https://github.com/fzyzcjy/triton.git \
    && cd triton && pip install --verbose -e .; \
  fi

# sgl_kernel cu130 for GB300
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    SGL_KERNEL_VERSION=0.3.17.post2 && \
    python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps; \
  fi

# TransformerEngine: cu13 无 wheel，从源码 build
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install nvidia-mathdx==26.6.0 && \
      pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.10; \
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.10.0"; \
    fi
```

还显式 `pip install "numpy<2"` —— 全栈版本是 Slime 团队预先调好的，不需要我们再爆破。

### Docker build 命令

```bash
cd /scratch/yuzhou/projects/ProRL-Agent-Server/slime
docker build \
  --build-arg ENABLE_CUDA_13=1 \
  --build-arg SGLANG_IMAGE_TAG=v0.5.12.post1-cu129 \
  --build-arg MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862 \
  --build-arg PATCH_VERSION=latest \
  -t slimerl/slime-cu130:local \
  -f docker/Dockerfile .
```

**注意：**
- `SGLANG_IMAGE_TAG` 用最新的 cu129 base（slimerl 没发 cu130 base，但 ENABLE_CUDA_13=1 会在 build 时 overlay cu130 sgl-kernel）。
- 预计 build 时间 **30-60min**（要 compile：flash-attn 2.7.4 + hopper FA + TransformerEngine + Apex + 可能 patched Triton）。
- 占用磁盘 60-80GB。我们这台 / 还有 300GB，OK。

### Build 完后跑训练

进容器：
```bash
docker run -it --rm \
  --gpus all --shm-size 32g \
  --ipc=host --network host \
  -v /scratch/yuzhou/projects/ProRL-Agent-Server:/workspace/ProRL-Agent-Server \
  -v /scratch/yuzhou/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  slimerl/slime-cu130:local \
  bash
```

容器内：
```bash
cd /workspace/ProRL-Agent-Server
# 转换权重（容器内 numpy/scipy/torch 都已对齐，应该一次过）
bash examples/swegym_slime_grpo/convert_weights.sh

# 启训练（注意 GPU/端口要根据机器调）
# 默认 8 GPU；我们这台 GPU 4,7 被别人占，只有 0,1,2,3 完全 free
TRAIN_GPUS=0,1 ROLLOUT_GPUS=2,3 \
ROLLOUT_NUM_GPUS=2 \
TRAIN_GPUS=0,1 \
SGLANG_ROUTER_PORT=19000 \
bash examples/swegym_slime_grpo/run.sh
```

（详细参数等 docker build 跑通 + 真的进容器跑了再补。）

## 4. 下一步 TODO

- [ ] docker build 完成 → 进容器验证 sglang/torch/numpy/scipy 全部 ok
- [ ] 进容器跑 `convert_weights.sh`（应该一次过）
- [ ] 改 `run.sh` 适配我们机器（GPU 0,1,2,3 only；ports 19xxx）
- [ ] 跑 1-2 步训练，看 grad_norm 和 reward 正常
- [ ] 把 docker build 命令固化成 fork 里的 `scripts/build_slime_cu130.sh`

## 5. 关键决定与背景

- **为什么 fork Polar 到 rucnyz**：要做 patch 适配 + 自定义 Dockerfile + 实验性 config，不适合上游 PR。和 `rucnyz/SkyRL` 同模式。
- **为什么用 Slime 而不是 SkyRL**：Polar 自带 Slime bridge `src/slime_bridge/`，端到端 example 在 `examples/swegym_slime_grpo/`。其他 trainer（NeMo-RL、VERL）在 Polar roadmap 上是 TODO。SkyRL bridge 完全没现成的。
- **为什么 PGC SWE 训练目前是 SkyRL（rucnyz/SkyRL `pgc-swe`）**：那是之前历史项目，已在 B300 上跑了一周，单独维护。Polar+Slime 这条线是 *新方向*（Claude Code / Codex as agent），跟 PGC 平行。
- **NeMo-RL 仓库（/scratch/yuzhou/projects/RL）**：分支 `qwen3.5-9b-swe` + 多个 `nemo_rl/` 未提交修改 —— 在等 Polar→NeMo-RL bridge 落地后再决定是否复活。目前先 stash，分支 push 到 origin 归档保留。

## 6. 故障排查参考

- 端口被占查询：`ss -tlnp '( sport = :8080 )'`
- nvidia-smi 看 GPU 占用：`nvidia-smi --query-gpu=index,name,memory.used --format=csv`
- 我们这台另一个 user 在用 GPU 4,5,6,7（DeepSeek-V4-Pro 等），别动他们的进程；GPU 0,1,2,3 是我们的
- 重启 SGLang docker 后 `_force_delete_via_rest` 不存在（那是 SkyRL/harbor 的 owner-reaper，不适用 Polar 这套）
- Slime 训练日志默认在 wandb；本地日志 `./logs/`
