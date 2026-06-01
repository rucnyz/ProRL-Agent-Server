#!/usr/bin/env bash
# Single source of truth for sequence-length knobs, sourced by BOTH
# run_host_polar.sh (host) and run_container_slime.sh (container).
#
# Keep the defaults here and nowhere else: the agent's output budget
# (CLAUDE_CODE_MAX_OUTPUT_TOKENS), slime's trained response/prompt caps
# (--rollout-max-response-len / --rollout-max-prompt-len) and the SGLang context
# window (--sglang-context-length) all derive from these, so they can't drift.
# Override any of them from the environment before launching.
#
#   ROLLOUT_MAX_RESPONSE_LEN  trained response cap == claude_code output budget
#   ROLLOUT_MAX_PROMPT_LEN    trained prompt cap
#   SGLANG_CONTEXT_LENGTH     inference KV window; must exceed the largest
#                             prompt+completion an agent actually sends. claude_code
#                             accumulates file contents / tool output and its prompt
#                             routinely grows past 50k tokens over a SWE session; a
#                             request whose prompt+completion overflows this window
#                             400s and (in bursts) trips the sgl-router circuit
#                             breaker. 96k holds prompt≈80k + ROLLOUT_MAX_RESPONSE_LEN
#                             comfortably; B300 (288GB) has ample KV headroom.
: "${ROLLOUT_MAX_RESPONSE_LEN:=16000}"
: "${ROLLOUT_MAX_PROMPT_LEN:=32000}"
# Qwen3.5-4B is natively 262144 (256K) and the SGLang KV pool already holds ~3.5M
# tokens, so use the full 256K — agentic sessions (esp. harbor terminal tasks) grow
# past 90k and a smaller cap 400s them. NOTE: run_container_slime.sh always sources
# THIS file (SCRIPT_DIR is hardcoded to swegym_slime_grpo), so this value governs the
# container's --sglang-context-length for harbor runs too.
: "${SGLANG_CONTEXT_LENGTH:=262144}"
