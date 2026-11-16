"""Select the same evaluate runs for aggregate tables and prediction analysis."""
from datetime import datetime, timezone
import json
import re
import sys


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def select_evaluation_runs(root, split=None):
    """Keep one non-debug run per experiment, seed and split, based on creation time."""
    selected = {}
    outputs = root / "outputs"
    for path in sorted(outputs.glob("*/evaluate/*/metrics.json")):
        run_dir = path.parent
        relative = run_dir.relative_to(root).as_posix()
        try:
            config = _read_json(run_dir / "config.json")
            if config.get("debug"):
                continue
            metrics = _read_json(path)
            run_split = metrics["split"]
            if split is not None and run_split != split:
                continue
            experiment = re.sub(r"-s\d+$", "", metrics["trained_experiment"])
            seed = int(metrics["trained_seed"])
            train_dir = (outputs / f"{experiment}-s{seed}" / "train").resolve()
            checkpoint = (root / metrics["checkpoint"]).resolve()
            if not train_dir.is_relative_to(outputs.resolve()) or not checkpoint.is_relative_to(train_dir):
                raise ValueError(f"checkpoint {checkpoint} is outside {train_dir}")
            if not checkpoint.is_file():
                raise ValueError(f"checkpoint {checkpoint} is missing")
            meta = _read_json(run_dir / "metadata.json")
            created = datetime.fromisoformat(meta["created_utc"].replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"WARNING: skipping {relative}: {exc}", file=sys.stderr)
            continue
        key = (experiment, seed, run_split)
        rank = (created, relative)
        run = {"experiment": experiment, "seed": seed, "split": run_split, "metrics": metrics,
               "run_dir": relative, "commit": meta.get("git_commit")}
        previous = selected.get(key)
        if previous is None:
            selected[key] = rank, run
            continue
        newer, older = sorted((previous, (rank, run)), key=lambda item: item[0], reverse=True)
        selected[key] = newer
        print(f"WARNING: duplicate {key}; skipping {older[1]['run_dir']} (keeping {newer[1]['run_dir']})",
              file=sys.stderr)
    return sorted((run for _, run in selected.values()), key=lambda run: run["run_dir"])
