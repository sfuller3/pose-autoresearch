#!/usr/bin/env python3
"""Render experiments/runs.jsonl as a sorted leaderboard.

Every training run appends a JSON record to the run log (see record_run in
train.py).  This script is the read side: it prints the table and refreshes
experiments/LEADERBOARD.md so a new session can see the full history at a
glance without re-reading raw logs.

Usage:
    python scripts/leaderboard.py            # print + write LEADERBOARD.md
    python scripts/leaderboard.py --top 10   # only the 10 best runs
    python scripts/leaderboard.py --no-write # print only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_LOG = REPO / "experiments" / "runs.jsonl"
LEADERBOARD = REPO / "experiments" / "LEADERBOARD.md"

# Column order for per-class accuracy; falls back to whatever keys exist.
CLASS_ORDER = [
    "fall", "eating", "working_together", "aggression",
    "unstable_gait", "wandering", "sitting_standing",
]
SHORT = {
    "fall": "fall", "eating": "eat", "working_together": "work",
    "aggression": "aggr", "unstable_gait": "gait", "wandering": "wand",
    "sitting_standing": "sit",
}


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    runs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn line from a killed process
    return runs


def render(runs: list[dict], top: int | None = None) -> str:
    if not runs:
        return "_No runs logged yet._\n"

    def fmt(v: float | None, places: int = 4) -> str:
        return "-" if v is None else f"{v:.{places}f}"

    ranked = sorted(runs, key=lambda r: r.get("val_acc") or 0, reverse=True)
    if top:
        ranked = ranked[:top]

    classes = [c for c in CLASS_ORDER if any(c in r.get("per_class", {}) for r in ranked)]
    header = ["#", "run", "val", "test"] + [SHORT.get(c, c) for c in classes]
    header += ["channels", "ep", "min", "params", "rev"]

    rows = []
    for i, r in enumerate(ranked, 1):
        pc = r.get("per_class", {})
        cfg = r.get("config", {})
        params = r.get("num_params")
        rows.append([
            str(i),
            r.get("run_name", "?"),
            fmt(r.get("val_acc")),
            fmt(r.get("test_acc")),
            *[fmt(pc.get(c), 3) for c in classes],
            "/".join(str(c) for c in cfg.get("block_channels", [])) or "-",
            str(r.get("epochs", "-")),
            f"{r.get('elapsed_s', 0) / 60:.0f}",
            f"{params / 1e6:.1f}M" if params else "-",
            r.get("git_rev", "-"),
        ])

    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]

    body = "\n".join(lines) + "\n"

    best = ranked[0]
    notes = [f"- **{r['run_name']}** — {r['notes']}" for r in ranked if r.get("notes")]
    out = [
        "# Run leaderboard",
        "",
        f"Generated from `experiments/runs.jsonl` ({len(runs)} run"
        f"{'s' if len(runs) != 1 else ''} logged). "
        "Regenerate with `python scripts/leaderboard.py`.",
        "",
        f"**Best so far:** `{best['run_name']}` — val {fmt(best.get('val_acc'))}, "
        f"test {fmt(best.get('test_acc'))}, checkpoint `{best.get('checkpoint', '?')}`",
        "",
        body,
    ]
    if notes:
        out += ["## Notes", "", *notes, ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=None, help="show only the N best runs")
    ap.add_argument("--no-write", action="store_true", help="print without updating LEADERBOARD.md")
    ap.add_argument("--log", type=Path, default=RUN_LOG, help="path to runs.jsonl")
    args = ap.parse_args()

    text = render(load_runs(args.log), top=args.top)
    print(text)

    if not args.no_write:
        LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
        LEADERBOARD.write_text(text)
        print(f"Wrote {LEADERBOARD.relative_to(REPO)}")


if __name__ == "__main__":
    main()
