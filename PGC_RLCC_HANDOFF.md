# PGC-RLCC Handoff（rucnyz fork, branch `pgc-rlcc`）

「在 Polar (NVIDIA NeMo) + Slime 上用 Claude Code / Codex 等真实 agent harness 跑 RL」的实验记录。本文档是给自己 / 同学复现用的，**会过时**；以本仓库实际代码为准。

最后更新时间：see `git log -1` on this branch。

## 0. 一句话现状

- ✅ **Polar 端到端 rollout smoke test 在 B300 上跑通**（reward 1.0，calculator example, claude_code harness, SGLang docker image）。
- ✅ **Slime cu130 docker image build 完成** (`slimerl/slime-cu130:local`, 73GB)，全栈在 B300 上可 import。
- ✅ **权重转换成功**（Qwen3.5-4B HF → Megatron torch_dist, 7.9G, 在容器内一次过）。
- ✅ **Slime+Polar 端到端训练管线在 B300 上跑起来了**（拆分编排：host 跑 Polar+apptainer，容器跑 Ray+Slime+SGLang，`--network host` 互通）。已验证：SGLang 2 引擎 ready、Megatron 加载、Polar gateway proxy、apptainer SWE-Gym sandbox 起来、agent 真的在解题（SGLang decode 300-500 tok/s、gateway `/v1/chat/completions` 200 OK）。
- ✅ **token-id blocker 已解决**：精简非流式 patch (`scripts/patch/patch_sglang_min.sh`) + `pi` 非流式 harness → `zero trainable tokens` 不再出现（整 run 0 次）。rollout/reward/advantage 链路全通，Megatron 加载 ckpt、pi agent 在 SWE-Gym sandbox 解题、SGLang decode 正常。
- ✅ **weight-sync NCCL hang 已解决**：`ROLLOUT_NUM_GPUS=1`（单引擎，`world_size=2`）→ weight update 2.0-2.2s 完成。2 引擎的 group fan-out 问题留待后面查。
- ✅✅ **首个真实 GRPO 训练步跑通（端到端闭环）**：`pi` harness + Qwen3.5-4B + 3-instance smoke。`step 1: train/grad_norm=2.94, pg_loss=-0.58, kl_loss=3e-4, tis=0.9997, global_batch_size=8`，checkpoint 落盘，权重同步回 SGLang（0.6s），干净 `Exit 0`（1 epoch smoke 数据耗尽）。链路：agentic rollout → swegym 评测 reward → GRPO advantage → Megatron 训练步 → ckpt → weight-sync。
- 🔧 **闭环路上修掉的两个 B300 专属坑**（都固化进脚本，见 §11）：
  1. **libcudart 冲突**：train actor 为 weight-sync `import sglang` 时，若 cu13 在其 `LD_LIBRARY_PATH` 上会把 `libcudart.so.13` 拉进进程，与 torch 的 `.so.12` 冲突，TE fused_attn 报 `Multiple libcudart libraries found`。解法：用 slime 自带 `--train-env-vars '{"LD_LIBRARY_PATH": "<不含 cu13>"}'` 给 train actor 单独设 cu13-free LD（Ray 实测完全覆盖、不 prepend）；SGLang 引擎仍走 job 级含 cu13 的 LD（它需要 sgl_kernel cu130）。
  2. **head_dim=256 无 attention 后端**：B300=sm103，TE 2.10 把 flash-attn head_dim>192 的白名单写死成 `(8,0)(9,0)(10,0)(12,0)`（漏了 sm103），cuDNN 9.16 fused 不支持 head_dim256，THD packing 又禁用 unfused → `No dot product attention backend is available`。解法：`scripts/patch/patch_te_sm103.sh` 给白名单加 `(10,3)`，flash-attn 2.7.4 的 kernel 实测能在 sm103+head_dim256+thd 上正确跑（sm103/B300 与 sm100/B200 同属 Blackwell，kernel 通用）。**注意：全 cu13 并不能解决这个 attention 问题（已验证），它和 cu12/cu13 正交。**
- 🚧 **下一步**：切 `claude_code` harness（流式）。需把 token-id emission 的流式分支 port 进 sglang patch（`patch_sglang_min.sh` 目前只覆盖非流式，配 `pi`）。详见 §8 路 A。

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

**用我们的封装脚本**（应用了下面列出的 4 个 patch，保证 B300 build 一次过）：

```bash
bash docker/slime-cu130/build.sh
```

或手动：
```bash
cp docker/slime-cu130/Dockerfile slime/docker/Dockerfile
cd slime
docker build \
  --build-arg ENABLE_CUDA_13=1 \
  --build-arg ENABLE_SGLANG_PATCH=0 \
  --build-arg SGLANG_IMAGE_TAG=v0.5.12.post1-cu129 \
  --build-arg MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862 \
  --build-arg PATCH_VERSION=latest \
  -t slimerl/slime-cu130:local \
  -f docker/Dockerfile .
```

### 我们对上游 Slime Dockerfile 做的 4 处修改（理由）

1. **`nvidia-mathdx==26.6.0` → `pybind11 nvidia-mathdx==25.6.0`**：上游 26.6.0 不在公开 pypi（最新公开 25.6.0）；同时 TE source build 需要 pybind11 但 base image 没装。
2. **`pip install -r /tmp/requirements.txt` → `apt-get remove -y python3-jwt; pip install --ignore-installed PyJWT; pip install -r /tmp/requirements.txt`**：base image 用 apt 装的 python3-jwt 2.7.0 没 RECORD 文件，pip 在装 ray[default] 的依赖时无法卸载。
3. **`sgl_kernel-0.3.17.post2+cu130` → `sglang_kernel-0.4.2.post2+cu130 + uninstall sgl-kernel`**：旧 sgl_kernel build against torch 2.9 ABI（symbol `c10_cuda_check_implementation(int, ...)`），但 image 自带 torch 2.11 ABI 是 `(unsigned int, ...)`，import 时 undefined symbol。新版 `sglang-kernel`（注意改名）的 cu130 wheel 是新 ABI，且 uninstall sgl-kernel 防止旧 abi3.so 被加载。
4. **`ENABLE_SGLANG_PATCH=0` 默认**（命令行传入，Dockerfile 上游有 `# TODO temporarily skip patching for GB200/GB300` 注释）：Slime 自带的 sglang.patch 与 sglang 0.5.12.post1 已经不兼容。

### 镜像启动 + 全栈 sanity 测试

```bash
docker run --rm --gpus '"device=0"' \
  -e LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/cuda/lib64 \
  slimerl/slime-cu130:local \
  python3 -c "
import torch, sglang, sgl_kernel, slime, megatron, transformer_engine.pytorch
print(torch.__version__, torch.cuda.is_available())
p = torch.cuda.get_device_properties(0)
print(f'{p.name} sm_{p.major}{p.minor}')
"
```

预期：`2.11.0+cu129 True` + `NVIDIA B300 SXM6 AC sm_103`。

**LD_LIBRARY_PATH 必须**——`sgl_kernel` 需要 `libnvrtc.so.13`（cu130 runtime），但 base image 默认 cu12 路径优先。

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

## 7. 端到端训练编排（拆分方案，已跑通管线）

容器缺 polar/apptainer/slime_bridge，host 缺干净的 slime 栈。所以拆分：

| 角色 | 跑在哪 | 起什么 |
|---|---|---|
| Polar rollout + gateway | **host** (.venv) | `examples/swegym_slime_grpo/run_host_polar.sh` |
| apptainer SWE-Gym sandbox | **host** (gateway 调起) | — |
| Ray + Slime train + SGLang engines | **容器** (`--network host`, GPU 0-3) | `examples/swegym_slime_grpo/run_container_slime.sh` |

`--network host` 让两边 127.0.0.1 / host-IP 共享网络：
- 容器 SGLang router 绑 host-IP:19000 ← gateway 必须用 host-IP（不是 127.0.0.1）连它，run_host_polar.sh 自动探测 host IP 写进 topology。
- slime_bridge(容器) → host Polar rollout :18080；gateway → 容器 SGLang :19000。

### 复现（权重已转好的前提下）

```bash
# 1. host 起 polar（后台）
cd /scratch/yuzhou/projects/ProRL-Agent-Server
mkdir -p tmp/swegym_slime_grpo
set -a; source /scratch/yuzhou/projects/RL/research/pgc_swe/.env; set +a   # E2B/HF keys
bash examples/swegym_slime_grpo/run_host_polar.sh > tmp/swegym_slime_grpo/host_polar.log 2>&1 &
# 等 health: curl -sf http://127.0.0.1:18080/health

# 2. 容器起训练（3-instance smoke 子集）
docker run -d --name slime-train \
  --gpus '"device=0,1,2,3"' --shm-size 32g --ipc=host --network host \
  -v /scratch/yuzhou/projects/ProRL-Agent-Server:/workspace/ProRL-Agent-Server \
  -v /scratch/yuzhou/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN=$HF_TOKEN \
  slimerl/slime-cu130:local \
  bash -lc "cd /workspace/ProRL-Agent-Server && \
    PROMPT_DATA=/workspace/ProRL-Agent-Server/examples/swegym_slime_grpo/swegym_train_smoke3.jsonl \
    ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=4 SAVE_INTERVAL=5 \
    bash examples/swegym_slime_grpo/run_container_slime.sh"
docker logs -f slime-train
```

端口（本机 8080/9000/6379 被另一 user 占）：Polar rollout 18080 / gateway 18100 / SGLang router 19000 / Ray GCS 6380 / Ray dashboard 8266。换机器可改回默认。

### 踩过的坑（已在脚本里修好）

1. **Ray GCS 6379 冲突**（`--network host` + 别人占了 6379）→ `RAY_GCS_PORT=6380 RAY_DASHBOARD_PORT=8266`（run_container_slime.sh 已参数化）。
2. **gateway 502 Bad Gateway / All connection attempts failed**：Slime SGLang router 绑 host-IP（get_host_info），gateway 用 127.0.0.1 连不上 → run_host_polar.sh 自动探测 host IP 写进 topology sglang.base_url。
3. **prompt-data 全 293 条但只 build 了 3 个 SIF** → `swegym_train_smoke3.jsonl`（3 instance 子集）。

## 8. ⛔ 剩余 blocker：SGLang token-id patch 适配 0.5.12.post1

现象：`Dropping Polar group N because of zero trainable tokens`。slime_bridge/adapter.py 因 trace 缺 prompt/response token_ids 把所有 sample drop（loss_mask 全 0）。根因：容器 SGLang 不吐 token_ids。

修复：把 Polar 的 `scripts/patch/patch_sglang.sh`（写给 0.5.10）适配到容器的 sglang 0.5.12.post1。已做了一半，存为 **`scripts/patch/patch_sglang_0512.sh`**：

| patch 目标文件 | 0.5.12.post1 适配状态 |
|---|---|
| `srt/entrypoints/openai/protocol.py`（schema: token_id/input_token_ids 字段） | ✅ 原样匹配 |
| `srt/entrypoints/openai/utils.py`（append token_id） | ✅ 原样匹配 |
| `srt/managers/tokenizer_manager.py` | ✅ 已改：snippet3 `total_retractions` → `num_retractions` |
| `serving_chat.py` 非流式 `ChatCompletionResponseChoice(...)` | ✅ 原样匹配 |
| `serving_chat.py` `ChatCompletionTokenLogprob` token_logprobs loop | ✅ 原样匹配 |
| `serving_chat.py` **streaming snippets**（_process_tool_call_stream 签名/调用、streaming choice_data、normal_text/tool_calls choice_data） | ⛔ **未适配** — 0.5.12 重构了这段（+591 行），snippet 对不上 |

### 两条往下走的路（二选一）

**路 A — port streaming snippets**：进容器 `sed -n` 看 `serving_chat.py` 的 streaming 段（`_process_tool_call_stream` 定义 + 各 `ChatCompletionResponseStreamChoice(` 构造点），把 patch_sglang_0512.sh 里对应 5 个 `replace_once` 的 `old` 块改成 0.5.12 的实际文本。改完 `docker run --rm -v .../patch_sglang_0512.sh:... slimerl/slime-cu130:local bash -lc 'bash /tmp/patch.sh'` 应打印 `Patched SGLang in ...`。然后把 patch bake 进镜像（docker/slime-cu130/Dockerfile 加一层 `RUN bash patch_sglang_0512.sh`）或容器启动时先跑。

**路 B — 最小 patch + 非流式 harness**（更省事）：非流式路径 5 个 snippet 已全部匹配。做一个只含 protocol+utils+tokenizer_manager+非流式choice+token_logprobs-loop 的精简 patch（去掉所有 streaming `replace_once`），然后把 `polar_config.yaml` 的 `agent.harness` 从 `qwen_code`（require_streaming=true）换成 **非流式 harness**（`openhands_sdk` 或 `pi`，require_streaming=false）。这样 gateway 走非流式响应，已 patch 的非流式代码就能吐 token_ids。代价：换了 agent harness（实验设定变化）。

适配/验证 patch 后，重跑第 7 节的复现命令，预期不再 `zero trainable tokens`，能看到 `grad_norm` / GRPO step。

## 9. ⛔ 当前 blocker：weight-sync NCCL hang（2-engine 配置）

### 现象
patch + pi harness 重跑后，前半全通：
- `zero trainable tokens` = **0 次**（token-id patch 生效）
- Megatron 加载 ckpt（iter 0, TP=2）、pi agent 在 sandbox 解题、SGLang decode 正常、gateway 200 OK
- 卡在首个 GRPO step 的 **weight update**（Megatron→SGLang 同步新权重）

日志关键：
```
(MegatronTrainRayActor) Timer update_weights start
(SGLangEngine pid=18323) init custom process group: ... rank=1, world_size=3, group_name=slime-pp_0, backend=nccl
(SGLangEngine pid=18323) POST /init_weights_update_group HTTP/1.1 200 OK
(SGLangEngine pid=18323) POST /pause_generation 200 OK
```
然后 25min+ 无进展，GPU 全 0%。`init_weights_update_group` 只被调用 **1 次**（期望 2，每 engine 一次）。engine 18322 还活着但从没 init group → `world_size=3` 的 NCCL group 永远等不齐 → broadcast 阻塞。

### 怀疑根因
slime 的 weight-update group 构造假设跟我们的 rollout 配置（`--rollout-num-gpus 2 --rollout-num-gpus-per-engine 1` = 2 个独立 TP=1 引擎）对不上。weight update group 应该把 train rank 0 + **所有** rollout engine 拉进同一个 `world_size = 1 + n_engines` 的 NCCL group，但只有一个 engine 收到 init 调用。

### 往下排查的方向（下次从这开始）
1. **先试单引擎**：`ROLLOUT_NUM_GPUS=1`（1 个 SGLang 引擎，world_size=2）。如果单引擎能过 weight sync → 确认是多引擎 group 编排问题，且单引擎可作为 smoke 的可行配置（GPU: 2 train + 1 rollout + 1 空）。**这是最快验证端到端训练能不能闭环的路。**
2. 若要 2 引擎：读 slime `slime/backends/sglang_utils` + `ray/` 里 weight update group 的构造（`init_weights_update_group` 是怎么 fan-out 给 engine 的），看是不是 `--network host` 下 engine 发现/编址有问题，或某个 `--sglang-*` flag 控制 group 成员。
3. NCCL 接口：`--network host` 下机器有多张网卡（bond0/ens*），可能需要 `NCCL_SOCKET_IFNAME=` 指定。但当前现象是「group 没建齐」而非「建了连不上」，所以先查 #1/#2。

### 复现到这一步
第 7 节命令照跑（patch + pi 都已固化）。weight sync hang 在首个 step（rollout 完成后）。

## 10. 进度全景（时间线）

| 阶段 | 状态 |
|---|---|
| Polar rollout smoke (claude_code, reward 1.0) | ✅ |
| Slime cu130 docker image (B300) | ✅ |
| 权重转换 HF→Megatron | ✅ |
| host/container 拆分编排 | ✅ |
| 端到端管线启动（引擎/Megatron/gateway/apptainer/agent 全活） | ✅ |
| token-id patch（非流式）+ pi harness → trainable tokens | ✅ |
| **weight-sync NCCL (2-engine)** | ⛔ §9 |
| 首个 GRPO grad_norm | ⏳ 待 weight-sync 解决 |
