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
#                             breaker. Qwen3.5-4B supports 262144 (256K) natively and
#                             the KV pool already holds ~3.5M tokens, so use the full
#                             256K — agentic harbor sessions can grow past 90k and a
#                             smaller cap (we earlier used 98304) makes them 400.
: "${ROLLOUT_MAX_RESPONSE_LEN:=16000}"
: "${ROLLOUT_MAX_PROMPT_LEN:=32000}"
: "${SGLANG_CONTEXT_LENGTH:=262144}"
