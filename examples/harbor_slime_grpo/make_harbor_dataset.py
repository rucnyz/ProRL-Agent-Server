#!/usr/bin/env python3
"""Convert NVIDIA Harbor terminal tasks into a Polar GRPO dataset (jsonl).

Each Harbor task dir (``<skill>/<skill>_task_NNNN/``) provides:
  - instruction.md         → the agent prompt
  - environment/files/      → files overlaid into /app at session start
  - tests/test.sh + test_outputs.py → the verifier (graded by harbor_verifier)

We emit one Polar row per task, in the same shape as the swegym dataset
(``prompt`` = chat messages, ``label``, ``metadata``):

    {
      "prompt": [{"role": "user", "content": "<instruction.md>"}],
      "label": "<skill>",
      "metadata": {
        "instance_id": "<skill>::<task_name>",
        "skill": "<skill>",
        "harbor_task_dir": "<abs host path to the task dir>",
        "harbor_sif": "<abs host path to hb__<skill>.sif>"
      }
    }

``harbor_task_dir`` is bind-mounted read-only into the sandbox at /harbor_task
(see polar_config), so prepare can overlay environment/files/ into /app and the
harbor_verifier evaluator can run tests/test.sh in the agent's runtime.

Only tasks whose skill SIF exists under --sif-dir are emitted (skill dir name
``data_processing`` maps to ``hb__data-processing.sif`` — underscores→dashes).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def skill_to_sif(skill: str, sif_dir: Path) -> Path:
    return sif_dir / f"hb__{skill.replace('_', '-')}.sif"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-root", type=Path, required=True,
                    help="skill_based/mixed root containing <skill>/<skill>_task_NNNN/")
    ap.add_argument("--sif-dir", type=Path, default=Path("/scratch/yuzhou/.cache/harbor-sif"))
    ap.add_argument("--out", type=Path, required=True, help="output jsonl path")
    ap.add_argument("--skills", nargs="*", default=None,
                    help="restrict to these skill dir names (default: all with a SIF)")
    ap.add_argument("--max-per-skill", type=int, default=None,
                    help="cap tasks per skill (smoke runs)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rows: list[dict] = []
    skipped_no_sif: list[str] = []
    skill_dirs = sorted(p for p in args.task_root.iterdir() if p.is_dir())
    for skill_dir in skill_dirs:
        skill = skill_dir.name
        if args.skills and skill not in args.skills:
            continue
        sif = skill_to_sif(skill, args.sif_dir)
        if not sif.is_file():
            skipped_no_sif.append(skill)
            continue
        tasks = sorted(t for t in skill_dir.iterdir()
                       if t.is_dir() and (t / "instruction.md").is_file()
                       and (t / "tests" / "test.sh").is_file())
        rng.shuffle(tasks)
        if args.max_per_skill:
            tasks = tasks[: args.max_per_skill]
        for t in tasks:
            instruction = (t / "instruction.md").read_text(errors="replace")
            rows.append({
                "prompt": [{"role": "user", "content": instruction}],
                "label": skill,
                "metadata": {
                    "instance_id": f"{skill}::{t.name}",
                    "skill": skill,
                    "harbor_task_dir": str(t.resolve()),
                    "harbor_sif": str(sif.resolve()),
                },
            })
    rng.shuffle(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by_skill: dict[str, int] = {}
    for r in rows:
        by_skill[r["label"]] = by_skill.get(r["label"], 0) + 1
    print(f"wrote {len(rows)} rows -> {args.out}")
    for s, n in sorted(by_skill.items()):
        print(f"  {s}: {n}")
    if skipped_no_sif:
        print(f"skipped skills without a SIF: {skipped_no_sif}")


if __name__ == "__main__":
    main()
