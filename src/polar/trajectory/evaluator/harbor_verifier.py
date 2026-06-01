"""``harbor_verifier`` evaluator — grade NVIDIA Harbor terminal tasks.

Use this strategy for Harbor (terminal-bench-style) tasks adapted into Polar.
A Harbor task ships, on the host, a directory with:

    <task>/instruction.md        # the prompt (→ Polar instruction)
    <task>/environment/files/    # files the agent works on (overlaid into WORKDIR)
    <task>/tests/test.sh         # the verifier entrypoint
    <task>/tests/test_outputs.py # pytest assertions on the agent's outputs

The whole task directory is bind-mounted read-only into the rollout sandbox
(see the harbor task template's ``runtime.kwargs.volumes`` —
``{sample.metadata.harbor_task_dir}:/harbor_task:ro``). The agent solves the
task in ``workdir`` (``/app`` by convention, matching the Harbor Dockerfiles'
``WORKDIR /app; COPY files/ /app/``); this evaluator then reproduces Harbor's
verifier contract *in the same runtime the agent used*:

    1. copy ``/harbor_task/tests/.`` → ``/tests/`` and make ``/logs/verifier``
    2. run ``bash /tests/test.sh`` from ``workdir``  (it pip-installs pytest +
       any test_requirements.txt, runs pytest, and writes the reward)
    3. read ``/logs/verifier/reward.txt`` (Harbor's contract: ``1`` = all tests
       pass, ``0`` = any fail) → ``outcome_reward``

Because grading runs in the agent's own runtime, the harbor task template MUST
set ``evaluator.refresh_runtime: false`` (do NOT spin a fresh sandbox — the
agent's filesystem changes are what we grade).

Config schema (``EvaluatorSpec.config``)
----------------------------------------
- ``task_mount`` *(str, default ``/harbor_task``)* — where the task dir is
  bind-mounted inside the runtime (its ``tests/`` subdir is used).
- ``workdir`` *(str, default ``/app``)* — directory the agent worked in; the
  verifier runs from here.
- ``test_timeout`` *(float, default 900)* — seconds allowed for ``test.sh``.
"""

from __future__ import annotations

import re
from typing import Any

from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory

_REWARD_RE = re.compile(r"HARBOR_REWARD=([-+0-9.eE]+)")


class HarborVerifierEvaluator(BaseTrajectoryEvaluator):
    """Grades Harbor terminal tasks by running their ``tests/test.sh``."""

    MODE = "harbor_verifier"

    def __init__(
        self,
        *,
        task_mount: str = "/harbor_task",
        workdir: str = "/app",
        test_timeout: float = 900.0,
        **_: Any,
    ) -> None:
        self.task_mount = task_mount.rstrip("/")
        self.workdir = workdir
        self.test_timeout = float(test_timeout)

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        rt = runtime.get("runtime")
        if rt is None:
            return EvalResult(outcome_reward=0.0, metadata={"harbor_error": "no runtime"})

        # Reproduce Harbor's verifier contract inside the agent's runtime.
        # test.sh refuses to run from "/", so cd into the workdir first; it also
        # writes /logs/verifier/reward.txt itself (echo 1 / echo 0).
        script = (
            "set +e; "
            "mkdir -p /tests /logs/verifier /logs/agent; "
            f"cp -a {self.task_mount}/tests/. /tests/ 2>/dev/null; "
            f"cd {self.workdir} 2>/dev/null || cd /root 2>/dev/null || cd /tmp; "
            "bash /tests/test.sh > /logs/verifier/test_stdout.txt 2>&1; "
            'echo "HARBOR_REWARD=$(cat /logs/verifier/reward.txt 2>/dev/null || echo 0)"'
        )
        try:
            res = await rt.exec(script, timeout_sec=self.test_timeout)
        except Exception as exc:  # noqa: BLE001 — never crash the gateway on grading
            return EvalResult(outcome_reward=0.0, metadata={"harbor_error": f"exec failed: {exc}"})

        stdout = res.stdout or ""
        reward = 0.0
        matches = _REWARD_RE.findall(stdout)
        if matches:
            try:
                reward = float(matches[-1])
            except ValueError:
                reward = 0.0
        # Clamp to [0, 1] — Harbor rewards are binary but be defensive.
        reward = max(0.0, min(1.0, reward))
        return EvalResult(
            outcome_reward=reward,
            metadata={
                "harbor_return_code": res.return_code,
                "harbor_reward_parsed": bool(matches),
            },
        )
